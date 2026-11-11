from types import SimpleNamespace

import torch

from config import DEVICE, RISK


class RiskCost:
    def __init__(self, om):
        self.lo, self.v, self.n = om.lo.clone(), float(om.v), tuple(int(x) for x in om.n)
        self.thresholds = [float(x) for x in RISK["thresholds"]]
        self.fgrid = torch.full((1, len(self.thresholds)) + self.n[::-1], 10.0, device=DEVICE)
        self.w = torch.zeros(1, device=DEVICE)

    def load(self, om, P, keep):
        unk = om.unknown() & ~keep
        for k, tau in enumerate(self.thresholds):
            m = unk & (P > tau)
            F = om.distance_field(m) if bool(m.any()) else torch.full(self.n, 10.0, device=DEVICE)
            self.fgrid[0, k].copy_(F.permute(2, 1, 0))
        torch.cuda.synchronize()

    def set_weight(self, w: float):
        self.w[0] = float(w)
        torch.cuda.synchronize()

    def _sample(self, X):
        u = (X - self.lo) / self.v - 0.5
        nx, ny, nz = self.n
        g = torch.stack([u[..., 0] / (nx - 1), u[..., 1] / (ny - 1), u[..., 2] / (nz - 1)], -1) * 2 - 1
        sh = g.shape[:-1]
        out = torch.nn.functional.grid_sample(self.fgrid, g.reshape(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(len(self.thresholds), -1).t().reshape(*sh, len(self.thresholds))

    @staticmethod
    def _act(pen, eta: float):
        pc = pen.clamp_min(0.0)
        return torch.where(pc < eta, 0.5 * pc * pc / eta, pc - 0.5 * eta)

    def penalty(self, state, stride: int) -> torch.Tensor:
        q = state.joint_state.position
        B, H = q.shape[0], q.shape[1]
        ih = torch.arange(0, H, stride, device=DEVICE)
        S = state.robot_spheres[:, ih]
        live = (S[..., 3] > 0).float()
        r = torch.where(S[..., 3] > 0, S[..., 3], torch.full_like(S[..., 3], -1.0))
        h = self._act(r[..., None] + RISK["eta"] - self._sample(S[..., :3]), RISK["eta"]) * live[..., None]
        return torch.zeros(B, H, device=DEVICE).index_copy(1, ih, h.sum((-1, -2)) * stride)

    def cost(self, state) -> torch.Tensor:
        return self.w[0] * self.penalty(state, RISK["stride"])

    def route(self, rig, qs, dt) -> float:
        if len(qs) < 4:
            qs = torch.cat([qs, qs[-1:].expand(4 - len(qs), -1)])
        S, _ = rig.spheres(qs)
        state = SimpleNamespace(joint_state=SimpleNamespace(position=qs[None]), robot_spheres=S[None])
        with torch.no_grad():
            c = float(self.penalty(state, 1).sum())
        return round(c * float(dt), 4)


def keep_mask(om, goal_xyz, arm_spheres):
    keep = torch.zeros(om.N, dtype=torch.bool, device=DEVICE)
    keep |= om.cells_near(torch.cat([goal_xyz.view(1, 3), torch.zeros(1, 1, device=DEVICE)], 1), RISK["goal_zone"])
    if arm_spheres is not None and len(arm_spheres):
        keep |= om.cells_near(arm_spheres, RISK["arm_band"])
    return keep


def install(solver, risk: RiskCost) -> int:
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager

    class RiskCostManager(RobotCostManager):
        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            cc = super().compute_costs(state, cost_collection, goal, **kwargs)
            val = self._risk.cost(state)
            ref = cc.values[0] if len(cc.values) else None
            if ref is not None and ref.dim() == 3 and val.dim() == 2:
                val = val.unsqueeze(-1)
            cc.add(val, "risk")
            return cc

    n = 0
    seen = set()
    for s in (solver, getattr(solver, "core", None)):
        for r in (getattr(s, "optimizer_rollouts", None) or []):
            cm = getattr(r, "cost_manager", None)
            if cm is None or id(cm) in seen:
                continue
            seen.add(id(cm))
            cm.__class__ = RiskCostManager
            cm._risk = risk
            n += 1
    return n
