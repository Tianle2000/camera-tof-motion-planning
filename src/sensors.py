import math

import numpy as np
import torch

from config import DEVICE, TOF_RANGE

TOF_ZONES = 8
TOF_FOV = math.radians(45.0)
TOF_FX = (TOF_ZONES / 2.0) / math.tan(TOF_FOV / 2.0)
TOF_CX = (TOF_ZONES - 1) / 2.0
TOF_TAN_HALF = math.tan(TOF_FOV / 2.0)
TOF_R_MIN, TOF_R_MAX = 0.02, 4.0
TOF_HZ = 15.0
TOF_SIGMA_REL, TOF_SIGMA_FLOOR = 0.03, 0.002
TOF_MERGE_SEP = 0.60
TOF_R_DET = 1.1
TOF_SUBRAYS = 8


def _zone_grid():
    idx = np.arange(TOF_ZONES)
    uu, vv = np.meshgrid(idx, idx)
    d = np.stack([(uu - TOF_CX) / TOF_FX, (vv - TOF_CX) / TOF_FX, np.ones_like(uu, dtype=float)], -1)
    dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
    uv = np.stack([uu, vv], -1).reshape(-1, 2).astype(float)
    return dirs, uv


TOF_ZONE_DIRS, TOF_ZONE_UV = _zone_grid()


def depth_sigma(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    s0 = 0.0012 + 0.0019 * (z - 0.4) ** 2
    th = theta.clamp(0.0, math.radians(89.0))
    obl = 0.0001 / z.clamp_min(1e-3).sqrt() * th ** 2 / (math.pi / 2.0 - th) ** 2
    return torch.where(th >= math.radians(60.0), s0 + obl, s0)


def ray_spheres_hit(org: torch.Tensor, dirs: torch.Tensor, sph: torch.Tensor, ignore: torch.Tensor = None):
    if len(sph) == 0:
        return torch.full((len(org),), float("inf"), device=org.device), torch.zeros_like(org)
    L = sph[None, :, :3] - org[:, None]
    tca = (L * dirs[:, None]).sum(-1)
    d2 = (L * L).sum(-1) - tca * tca
    r2 = sph[None, :, 3] ** 2
    thc = (r2 - d2).clamp_min(0).sqrt()
    inside = (L * L).sum(-1) <= r2
    t = torch.where(inside, torch.zeros_like(tca), torch.where((d2 <= r2) & (tca - thc > 0), tca - thc, torch.full_like(tca, float("inf"))))
    if ignore is not None:
        t = torch.where(ignore, torch.full_like(t, float("inf")), t)
    tk, k = t.min(1)
    p = org + torch.nan_to_num(tk, posinf=0.0)[:, None] * dirs
    n = torch.nn.functional.normalize(p - sph[k, :3], dim=1)
    n = torch.where((torch.isfinite(tk) & (tk > 0))[:, None], n, torch.zeros_like(n))
    return tk, n


def ray_spheres_t(org: torch.Tensor, dirs: torch.Tensor, sph: torch.Tensor) -> torch.Tensor:
    if len(sph) == 0:
        return torch.full((len(org),), float("inf"), device=org.device)
    L = sph[None, :, :3] - org[:, None]
    tca = (L * dirs[:, None]).sum(-1)
    d2 = (L * L).sum(-1) - tca * tca
    r2 = sph[None, :, 3] ** 2
    thc = (r2 - d2).clamp_min(0).sqrt()
    inside = (L * L).sum(-1) <= r2
    t = torch.where(inside, torch.zeros_like(tca), torch.where((d2 <= r2) & (tca - thc > 0), tca - thc, torch.full_like(tca, float("inf"))))
    return t.amin(1)


class Camera:
    def __init__(self, pos, R_cam_to_world, w=160, h=120, f=200.0, zmin=0.05, zmax=5.0, seg_dist=0.05):
        self.pos = torch.as_tensor(pos, dtype=torch.float32, device=DEVICE)
        self.R = torch.as_tensor(R_cam_to_world, dtype=torch.float32, device=DEVICE)
        self.w, self.h, self.f, self.zmin, self.zmax, self.seg_dist = w, h, f, zmin, zmax, seg_dist
        self.cx, self.cy = w / 2.0, h / 2.0
        uu, vv = torch.meshgrid(torch.arange(w, device=DEVICE) + 0.5, torch.arange(h, device=DEVICE) + 0.5, indexing="xy")
        dc = torch.stack([(uu - self.cx) / f, (vv - self.cy) / f, torch.ones_like(uu)], -1).reshape(-1, 3)
        self.cosz = 1.0 / dc.norm(dim=1)
        self.dirs = torch.nn.functional.normalize(dc, dim=1) @ self.R.T
        self.K = torch.tensor([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1.0]], device=DEVICE)

    def render(self, scene, spheres: torch.Tensor, gen: torch.Generator):
        org = self.pos.expand_as(self.dirs)
        t_obs, n_obs = scene.raycast(org, self.dirs)
        live = spheres[spheres[:, 3] > 0]
        t_rob = ray_spheres_t(org, self.dirs, live)
        robot_first = t_rob < t_obs
        t = torch.minimum(t_obs, t_rob)
        cos_inc = (-(self.dirs * n_obs).sum(1)).clamp(-1, 1).abs()
        theta = torch.where(robot_first, torch.zeros_like(cos_inc), torch.arccos(cos_inc))
        z = t * self.cosz
        noise = torch.randn(z.shape, generator=gen, device=DEVICE) * depth_sigma(z.clamp(self.zmin, self.zmax), theta)
        zn = z + noise
        valid = torch.isfinite(t) & (zn >= self.zmin) & (zn <= self.zmax)
        p = self.pos + self.dirs * (zn / self.cosz)[:, None]
        dmin = torch.cat([(torch.cdist(p[k:k + 8192], live[:, :3]) - live[None, :, 3]).amin(1) for k in range(0, len(p), 8192)])
        seg = dmin < self.seg_dist
        depth = torch.where(valid & ~seg, zn, torch.zeros_like(zn))
        raw = torch.where(valid, zn, torch.zeros_like(zn))
        return depth.view(self.h, self.w), raw.view(self.h, self.w)


class ToFArray:
    def __init__(self, mounts, kin, self_occlusion: np.ndarray, seed: int = 0):
        self.kin = kin
        self.links = sorted({m[0] for m in mounts})
        self.mount_link = [m[0] for m in mounts]
        self.pos = torch.as_tensor(np.array([m[1] for m in mounts]), dtype=torch.float32, device=DEVICE)
        self.rot = torch.as_tensor(np.array([m[2] for m in mounts]), dtype=torch.float32, device=DEVICE)
        self.K = len(mounts)
        self.zdirs = torch.as_tensor(TOF_ZONE_DIRS, dtype=torch.float32, device=DEVICE)
        self.zuv = torch.as_tensor(TOF_ZONE_UV, dtype=torch.float32, device=DEVICE)
        names = list(kin.kinematics_config.all_link_names)
        sph_link = kin.kinematics_config.link_sphere_idx_map.to(DEVICE)
        mli = torch.tensor([names.index(l) for l in self.mount_link], device=DEVICE)
        self.ignore_by_module = mli[:, None] == sph_link[None, :]
        self.self_occlusion = torch.as_tensor(self_occlusion, dtype=torch.float32, device=DEVICE)
        rng = np.random.default_rng(seed)
        self.phase = torch.as_tensor(rng.uniform(0, 1.0 / TOF_HZ, self.K), dtype=torch.float32, device=DEVICE)
        self.last_frame = torch.full((self.K,), -1, dtype=torch.long, device=DEVICE)

    def due(self, t: float) -> torch.Tensor:
        fr = torch.floor((t - self.phase) * TOF_HZ).long()
        new = (fr > self.last_frame) & (fr >= 0)
        self.last_frame = torch.where(new, fr, self.last_frame)
        return new

    def reset(self):
        self.last_frame.fill_(-1)

    def poses(self, q: torch.Tensor):
        from curobo._src.types.pose import Pose
        pp = self.kin.get_link_poses(q.view(1, -1), self.links)
        P = pp.position.view(-1, 3); Rl = Pose(position=P, quaternion=pp.quaternion.view(-1, 4)).get_rotation_matrix().view(-1, 3, 3)
        li = torch.tensor([self.links.index(l) for l in self.mount_link], device=DEVICE)
        S = P[li] + torch.einsum("kij,kj->ki", Rl[li], self.pos)
        R = torch.einsum("kij,kjl->kil", Rl[li], self.rot)
        return S, R

    def spheres(self, q: torch.Tensor) -> torch.Tensor:
        from curobo._src.state.state_joint import JointState
        return self.kin.compute_kinematics(JointState.from_position(q.view(1, -1))).robot_spheres.view(-1, 4).clone()

    def sub_dirs_local(self, n: int, gen: torch.Generator = None) -> torch.Tensor:
        k = torch.arange(n, device=DEVICE, dtype=torch.float32)
        a, b = k.repeat(n), k.repeat_interleave(n)
        if gen is None:
            ju = jv = torch.full((64, n * n), 0.5, device=DEVICE)
        else:
            ju = torch.rand((64, n * n), generator=gen, device=DEVICE); jv = torch.rand((64, n * n), generator=gen, device=DEVICE)
        u = self.zuv[:, :1] - 0.5 + (a[None] + ju) / n
        v = self.zuv[:, 1:] - 0.5 + (b[None] + jv) / n
        d = torch.stack([(u - TOF_CX) / TOF_FX, (v - TOF_CX) / TOF_FX, torch.ones_like(u)], -1)
        return torch.nn.functional.normalize(d, dim=-1)

    def read(self, q: torch.Tensor, scene, active: torch.Tensor, gen: torch.Generator):
        idx = torch.nonzero(active).flatten()
        if len(idx) == 0:
            return None
        S, R = self.poses(q)
        S, R = S[idx], R[idx]
        M = len(idx)
        out = dict(org=S.repeat_interleave(64, 0), R=R.repeat_interleave(64, 0), zone=torch.arange(64, device=DEVICE).repeat(M),
                   module=idx.repeat_interleave(64))
        spheres = self.spheres(q)
        live = spheres[:, 3] > 0
        out["rng"] = self._read_cone(S, R, scene, spheres[live], self.ignore_by_module[idx][:, live], gen, idx)
        return out

    def _read_cone(self, S, R, scene, sph, ign, gen, idx):
        n = TOF_SUBRAYS; P = n * n; M = len(S); Z = M * 64
        dirs = torch.einsum("mij,zpj->mzpi", R, self.sub_dirs_local(n, gen)).reshape(-1, 3)
        org = S.repeat_interleave(64 * P, 0)
        ignr = ign.repeat_interleave(64 * P, 0)
        t = torch.empty(Z * P, device=DEVICE); cos = torch.empty(Z * P, device=DEVICE)
        rob = torch.empty(Z * P, dtype=torch.bool, device=DEVICE)
        CH = 1 << 16
        for a in range(0, Z * P, CH):
            sl = slice(a, a + CH)
            to, no = scene.raycast(org[sl], dirs[sl])
            tr, nr = ray_spheres_hit(org[sl], dirs[sl], sph, ignr[sl])
            r = tr < to
            t[sl] = torch.minimum(to, tr); rob[sl] = r
            cos[sl] = (dirs[sl] * torch.where(r[:, None], nr, no)).sum(1).abs()
        t, cos, rob = t.view(Z, P), cos.view(Z, P), rob.view(Z, P)
        t_own = self.self_occlusion[idx].reshape(Z, P)
        r_own = t_own < t
        t = torch.where(r_own, t_own, t); rob = rob | r_own; cos = torch.where(r_own, torch.ones_like(cos), cos)
        INF = float("inf")
        ret = torch.isfinite(t) & (t <= TOF_R_MAX)
        tz = torch.where(ret, t, torch.full_like(t, INF))
        s = torch.where(ret, cos / tz.clamp_min(1e-3) ** 2, torch.zeros_like(t)) / P
        thr = 1.0 / TOF_R_DET ** 2
        sel_d = torch.full((Z,), INF, device=DEVICE); sel_start = sel_d.clone()
        found = torch.zeros(Z, dtype=torch.bool, device=DEVICE)
        left = ret.clone()
        for _ in range(int(math.ceil(TOF_R_MAX / TOF_MERGE_SEP)) + 1):
            start = torch.where(left, tz, torch.full_like(tz, INF)).amin(1)
            if not bool(torch.isfinite(start).any()):
                break
            mem = left & (tz < start[:, None] + TOF_MERGE_SEP)
            Sw = (s * mem).sum(1)
            take = torch.isfinite(start) & (Sw >= thr) & ~found
            sel_d = torch.where(take, start, sel_d); sel_start = torch.where(take, start, sel_start)
            found |= take
            left &= ~mem
        t_rob = torch.where(rob & ret, tz, torch.full_like(tz, INF)).amin(1)
        lim = torch.where(found, sel_start + TOF_MERGE_SEP, torch.full_like(sel_start, TOF_RANGE))
        robot_bad = t_rob < lim
        sigma = torch.clamp(TOF_SIGMA_REL * torch.where(found, sel_d, torch.zeros_like(sel_d)), min=TOF_SIGMA_FLOOR)
        tn = torch.where(found, sel_d + torch.randn((Z,), generator=gen, device=DEVICE) * sigma,torch.full_like(sel_d, INF))
        no_echo = ~found | (tn > TOF_RANGE)
        valid = ~no_echo & (tn >= TOF_R_MIN)
        rng = torch.where(valid, tn, torch.full_like(tn, float("nan")))
        rng = torch.where(no_echo, torch.full_like(rng, INF), rng)
        return torch.where(robot_bad, torch.full_like(rng, float("nan")), rng)
