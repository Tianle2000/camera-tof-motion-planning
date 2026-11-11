import math

import numpy as np
import torch

from config import CAMERA_HZ, MOVING
from sensors import depth_sigma


class MovingObstacleDetector:
    def __init__(self, camera):
        self.p = dict(MOVING)
        self.cam = camera
        self.H, self.W = camera.h, camera.w
        self.dev = camera.pos.device
        self.dirs = camera.dirs.view(self.H, self.W, 3)
        self.cosz = camera.cosz.view(self.H, self.W)
        z = torch.zeros(self.H, self.W, device=self.dev)
        self.B, self.S, self.N, self.F = z.clone(), z.clone(), z.clone(), z.clone()
        self.t_last = None
        self.tracks = []

    @torch.no_grad()
    def prewarm(self, render, seed: int):
        gen = torch.Generator(device=self.dev); gen.manual_seed(seed + 104729)
        for k in range(int(self.p["prewarm"]), 0, -1):
            self.camera_frame(-k / CAMERA_HZ, render(gen))

    @torch.no_grad()
    def camera_frame(self, t: float, depth: torch.Tensor):
        p = self.p
        dt = 1.0 / 30.0 if self.t_last is None else max(t - self.t_last, 1e-3)
        self.t_last = t
        z = depth
        valid = z > 0
        B, S, N = self.B, self.S, self.N
        sig0 = depth_sigma(torch.where(B > 0, B, torch.full_like(B, 1.0)).clamp(0.05, 5.0), torch.zeros_like(B))
        thr = torch.clamp(p["k_sig"] * torch.maximum(sig0, 1.25 * S), min=p["min_diff"])
        seen = N >= p["n_warm"]
        fg = valid & seen & (z < B - thr)
        far = valid & (N >= 1) & (z > B + thr)
        new = valid & (N < 1)
        upd = valid & ~fg & ~far & ~new
        sig_z = depth_sigma(z.clamp(0.05, 5.0), torch.zeros_like(z))
        B_new = torch.where(new | far, z, torch.where(upd, B + p["alpha"] * (z - B), B))
        S_new = torch.where(new | far, 0.8 * sig_z, torch.where(upd, S + p["alpha"] * ((z - B).abs() - S), S))
        N_new = torch.where(new | far, torch.ones_like(N), torch.where(upd, N + 1, N))
        F_new = torch.where(fg, self.F + dt, torch.zeros_like(self.F))
        absorb = fg & (F_new >= p["absorb_s"])
        B_new = torch.where(absorb, z, B_new); N_new = torch.where(absorb, torch.ones_like(N), N_new); F_new = torch.where(absorb, torch.zeros_like(F_new), F_new)
        self.B, self.S, self.N, self.F = B_new, S_new, N_new, F_new
        fg = fg & ~absorb
        dets = self._blobs(fg, z) if int(fg.sum()) >= p["n_min"] else []
        self._track(t, dets)

    def _blobs(self, fg, z):
        from scipy import ndimage
        m = fg.cpu().numpy()
        lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel())
        out = []
        lab_t = torch.as_tensor(lab, device=self.dev)
        for k in np.nonzero(sizes >= self.p["n_min"])[0]:
            if k == 0:
                continue
            sel = lab_t == int(k)
            out.append(self.cam.pos + self.dirs[sel] * (z[sel] / self.cosz[sel])[:, None])
        return out

    def _subsample(self, P):
        return P[torch.linspace(0, len(P) - 1, min(len(P), self.p["max_pts"]), device=self.dev).long()]

    def _track(self, t, dets):
        p = self.p
        used = set()
        for tr in self.tracks:
            gap = t - tr["last_t"]
            pred = tr["c"] + tr["v"] * gap
            best, bd = None, p["gate"] + float(tr["v"].norm()) * gap
            for j, P in enumerate(dets):
                if j in used:
                    continue
                d = float((P.mean(0) - pred).norm())
                if d < bd:
                    best, bd = j, d
            if best is not None:
                used.add(best)
                P = dets[best]
                tr["hist"].append((t, P.mean(0)))
                tr["hist"] = tr["hist"][-p["hist"]:]
                tr["c"] = P.mean(0); tr["last_t"] = t; tr["hits"] += 1
                tr["pts"] = self._subsample(P)
                if len(tr["hist"]) >= 3:
                    ts = torch.tensor([h[0] for h in tr["hist"]], device=self.dev); cs = torch.stack([h[1] for h in tr["hist"]])
                    tm = ts - ts.mean()
                    tr["v"] = (tm[:, None] * (cs - cs.mean(0))).sum(0) / (tm * tm).sum().clamp_min(1e-9)
                if not tr["conf"] and tr["hits"] >= p["conf_hits"] and float(tr["v"].norm()) >= p["v_min"]:
                    tr["conf"] = True
        for j, P in enumerate(dets):
            if j in used:
                continue
            self.tracks.append(dict(c=P.mean(0), v=torch.zeros(3, device=self.dev), last_t=t, hits=1, conf=False,
                                    hist=[(t, P.mean(0))], pts=self._subsample(P)))
        self.tracks = [tr for tr in self.tracks if t - tr["last_t"] <= p["keep_s"]]

    def _confirmed(self, t: float):
        return [tr for tr in self.tracks if tr["conf"] and t - tr["last_t"] <= self.p["keep_s"]]

    def active(self, t: float) -> bool:
        return len(self._confirmed(t)) > 0

    def has_targets(self) -> bool:
        return len(self.tracks) > 0

    @torch.no_grad()
    def swept_mask(self, om, t: float):
        p = self.p
        live = self._confirmed(t)
        if not live:
            return None
        mask = torch.zeros(om.N, dtype=torch.bool, device=self.dev)
        taus = torch.arange(0.0, p["horizon"] + 1e-9, p["h_step"], device=self.dev)
        ext = torch.linspace(0.0, p["extrude"], max(2, int(math.ceil(p["extrude"] / om.v)) + 1), device=self.dev)
        for tr in live:
            P = tr["pts"]
            u = torch.nn.functional.normalize(P - self.cam.pos, dim=1)
            Pe = (P[:, None, :] + u[:, None, :] * ext[None, :, None]).reshape(-1, 3)
            sh = tr["v"][None] * (t - tr["last_t"] + taus)[:, None]
            X = (Pe[None] + sh[:, None]).reshape(-1, 3)
            f, ok = om.index(X)
            mask[f[ok]] = True
        return mask
