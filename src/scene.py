from copy import deepcopy

import numpy as np
import torch

from config import DEVICE

MOVING_KEY = "__moving__"


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class Scene:
    def __init__(self, obstacles: dict):
        cub = list(obstacles.get("cuboid", {}).values())
        cyl = list(obstacles.get("cylinder", {}).values())
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        self.box_c = f(np.array([c["pose"][:3] for c in cub], float).reshape(-1, 3))
        self.box_R = f(quat_wxyz_to_R(np.array([c["pose"][3:] for c in cub], float).reshape(-1, 4)).reshape(-1, 3, 3))
        self.box_h = f(np.array([c["dims"] for c in cub], float).reshape(-1, 3) / 2.0)
        self.cyl_c = f(np.array([c["pose"][:3] for c in cyl], float).reshape(-1, 3))
        self.cyl_R = f(quat_wxyz_to_R(np.array([c["pose"][3:] for c in cyl], float).reshape(-1, 4)).reshape(-1, 3, 3))
        self.cyl_r = f(np.array([c["radius"] for c in cyl], float))
        self.cyl_hh = f(np.array([c["height"] for c in cyl], float) / 2.0)

    def sdf(self, p):
        out = torch.full((len(p),), float("inf"), device=p.device)
        if len(self.box_c):
            q = (torch.einsum("nj,bji->nbi", p, self.box_R) - torch.einsum("bj,bji->bi", self.box_c, self.box_R)[None]).abs() - self.box_h[None]
            d = q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
            out = torch.minimum(out, d.amin(1))
        if len(self.cyl_c):
            l = torch.einsum("nj,cji->nci", p, self.cyl_R) - torch.einsum("cj,cji->ci", self.cyl_c, self.cyl_R)[None]
            dr = l[..., :2].norm(dim=-1) - self.cyl_r[None]; dz = l[..., 2].abs() - self.cyl_hh[None]
            dd = torch.stack([dr, dz], -1)
            d = dd.clamp_min(0).norm(dim=-1) + dd.amax(-1).clamp_max(0)
            out = torch.minimum(out, d.amin(1))
        return out

    def raycast(self, org, dirs):
        N = len(org); dev = org.device
        t = torch.full((N,), float("inf"), device=dev); n = torch.zeros((N, 3), device=dev)
        rr = torch.arange(N, device=dev)
        if len(self.box_c):
            o = torch.einsum("rj,bji->rbi", org, self.box_R) - torch.einsum("bj,bji->bi", self.box_c, self.box_R)[None]
            d = torch.einsum("rj,bji->rbi", dirs, self.box_R)
            h = self.box_h[None]
            inv = 1.0 / d
            t1 = (-h - o) * inv; t2 = (h - o) * inv
            bad = torch.isnan(t1) | torch.isnan(t2)
            tmin = torch.where(bad, torch.full_like(t1, -float("inf")), torch.minimum(t1, t2))
            tmax = torch.where(bad, torch.full_like(t1, float("inf")), torch.maximum(t1, t2))
            par_out = (d == 0) & (o.abs() > h)
            tn = tmin.amax(-1); tf = tmax.amin(-1)
            hit = (tn <= tf) & (tf > 0) & ~par_out.any(-1)
            tb = torch.where(hit, tn.clamp_min(0.0), torch.full_like(tn, float("inf")))
            tk, k = tb.min(1)
            ax = tmin[rr, k].argmax(-1)
            nl = torch.zeros((N, 3), device=dev)
            nl[rr, ax] = -torch.sign(d[rr, k, ax])
            nw = torch.einsum("rij,rj->ri", self.box_R[k], nl)
            better = tk < t
            t = torch.where(better, tk, t); n = torch.where(better[:, None], nw, n)
        if len(self.cyl_c):
            o = torch.einsum("rj,cji->rci", org, self.cyl_R) - torch.einsum("cj,cji->ci", self.cyl_c, self.cyl_R)[None]
            d = torch.einsum("rj,cji->rci", dirs, self.cyl_R)
            r = self.cyl_r[None]; hh = self.cyl_hh[None]
            a = d[..., 0] ** 2 + d[..., 1] ** 2
            b = 2 * (o[..., 0] * d[..., 0] + o[..., 1] * d[..., 1])
            c = o[..., 0] ** 2 + o[..., 1] ** 2 - r * r
            disc = b * b - 4 * a * c
            inf = torch.full_like(a, float("inf"))
            ts = torch.where((a > 0) & (disc >= 0), (-b - disc.clamp_min(0).sqrt()) / (2 * a), inf)
            zs = o[..., 2] + ts * d[..., 2]
            ts = torch.where((ts > 0) & (zs.abs() <= hh), ts, inf)
            caps = []
            for s in (-1.0, 1.0):
                tc = (s * hh - o[..., 2]) / d[..., 2]
                x = o[..., 0] + tc * d[..., 0]; y = o[..., 1] + tc * d[..., 1]
                caps.append(torch.where((tc > 0) & (x * x + y * y <= r * r) & torch.isfinite(tc), tc, inf))
            allt = torch.stack([ts, caps[0], caps[1]], -1)
            tcyl, which = allt.min(-1)
            inside = (c <= 0) & (o[..., 2].abs() <= hh)
            tcyl = torch.where(inside, torch.zeros_like(tcyl), tcyl)
            tk, k = tcyl.min(1); wk = which[rr, k]
            p = o[rr, k] + torch.where(torch.isfinite(tk), tk, torch.zeros_like(tk))[:, None] * d[rr, k]
            nl = torch.zeros((N, 3), device=dev)
            side = wk == 0
            nl[side, :2] = torch.nn.functional.normalize(p[side, :2], dim=1)
            nl[wk == 1, 2] = -1.0; nl[wk == 2, 2] = 1.0
            nw = torch.einsum("rij,rj->ri", self.cyl_R[k], nl)
            better = tk < t
            t = torch.where(better, tk, t); n = torch.where(better[:, None], nw, n)
        return t, n


def with_moving_box(p: dict) -> dict:
    m = p["moving"]
    o = deepcopy(p["obstacles"])
    o.setdefault("cuboid", {})[MOVING_KEY] = {"dims": list(m["dims"]), "pose": list(m["p0"]) + list(m["quat"])}
    return o


def moving_center_at(m: dict, t: float) -> list:
    t0 = float(m.get("t_beg", 0.0))
    s = min(max(float(t), t0), float(m["t_end"])) - t0
    return [float(a) + float(v) * s for a, v in zip(m["p0"], m["vel"])]


class MovingBox:
    def __init__(self, p: dict, obs_truth: dict, obs_render: dict, truth: Scene, render: Scene):
        self.m = p["moving"]
        self.scenes = [(truth, list(obs_truth["cuboid"]).index(MOVING_KEY)), (render, list(obs_render["cuboid"]).index(MOVING_KEY))]
        self.R = truth.box_R[self.scenes[0][1]].clone()
        self.h = truth.box_h[self.scenes[0][1]].clone()
        self.at(0.0)

    def at(self, t: float):
        self.c = torch.as_tensor(moving_center_at(self.m, t), dtype=torch.float32, device=DEVICE)
        for scene, k in self.scenes:
            scene.box_c[k] = self.c

    def sdf(self, pts: torch.Tensor) -> torch.Tensor:
        q = ((pts - self.c) @ self.R).abs() - self.h
        return q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
