import math

import torch

from config import DEVICE, MAP, TOF_RANGE
from sensors import TOF_SIGMA_FLOOR, TOF_SIGMA_REL, depth_sigma

CHUNK = 6_000_000


class OccupancyMap:
    def __init__(self, center, extent, voxel: float):
        self.v = float(voxel)
        self.p = dict(MAP)
        self.n = tuple(math.ceil(e / voxel - 1e-9) for e in extent)
        self.nt = torch.tensor(self.n, device=DEVICE)
        self.N = self.n[0] * self.n[1] * self.n[2]
        self.center = torch.as_tensor(center, dtype=torch.float32, device=DEVICE)
        self.lo = self.center - self.nt.float() * self.v / 2
        self.hi = self.lo + self.nt.float() * self.v
        self.L = torch.zeros(self.n, dtype=torch.float32, device=DEVICE)
        self._occ_max = torch.zeros(self.N, dtype=torch.float32, device=DEVICE)
        self._free = torch.zeros(self.N, dtype=torch.bool, device=DEVICE)
        self._edt = None
        self._spheres = None; self._robot_hit = None

    def reset(self):
        self.L.zero_()

    def index(self, p: torch.Tensor):
        ijk = torch.floor((p - self.lo) / self.v).long()
        ok = ((ijk >= 0) & (ijk < self.nt)).all(-1)
        flat = (ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2]
        return flat, ok

    def unknown(self) -> torch.Tensor:
        L = self.L.view(-1)
        return (L >= -self.p["th_free"]) & (L <= self.p["th_occ"])

    def _clip(self, org, dirs, t0, t1):
        inv = 1.0 / torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)
        a = (self.lo - org) * inv; b = (self.hi - org) * inv
        tmin = torch.minimum(a, b).amax(-1); tmax = torch.maximum(a, b).amin(-1)
        return torch.maximum(t0, tmin), torch.minimum(t1, tmax)

    def _march(self, org, dirs, t0, t1, with_t: bool = False):
        t0, t1 = self._clip(org, dirs, t0, t1)
        keep = t1 > t0
        rid = torch.nonzero(keep).flatten()
        if len(rid) == 0:
            e = torch.zeros(0, dtype=torch.long, device=DEVICE)
            return (e, e, torch.zeros(0, device=DEVICE)) if with_t else (e, e)
        org, dirs, t0, t1 = org[rid], dirs[rid], t0[rid], t1[rid]
        M = int(torch.ceil(((t1 - t0)[:, None] * dirs.abs()).max() / self.v).item()) + 2
        out_f, out_r, out_t = [], [], []
        per = max(1, CHUNK // (3 * M + 2))
        k = torch.arange(M, device=DEVICE, dtype=torch.float32)
        for a in range(0, len(rid), per):
            sl = slice(a, a + per)
            o, dv, a0, a1 = org[sl], dirs[sl], t0[sl], t1[sl]
            p0 = o + a0[:, None] * dv
            T = torch.empty((len(o), 3 * M + 2), device=DEVICE)
            T[:, 0] = a0; T[:, 1] = a1
            for ax in range(3):
                d_ = dv[:, ax]
                i0 = torch.floor((p0[:, ax] - self.lo[ax]) / self.v)
                first = torch.where(d_ > 0, i0 + 1, i0)
                plane = self.lo[ax] + (first[:, None] + torch.where(d_ > 0, 1.0, -1.0)[:, None] * k[None]) * self.v
                t = (plane - o[:, ax, None]) / torch.where(d_.abs() < 1e-12, torch.full_like(d_, 1e-12), d_)[:, None]
                torch.where((d_.abs()[:, None] < 1e-12) | (t <= a0[:, None]) | (t >= a1[:, None]), torch.full_like(t, float("inf")), t,
                            out=T[:, 2 + ax * M:2 + (ax + 1) * M])
            T = torch.sort(T, dim=1).values
            lo_, hi_ = T[:, :-1], T[:, 1:]
            val = torch.isfinite(hi_) & (hi_ - lo_ > 1e-9)
            vr, vc = torch.nonzero(val).unbind(1)
            mid = (lo_[vr, vc] + hi_[vr, vc]) / 2
            pts = o[vr] + mid[:, None] * dv[vr]
            f, ok = self.index(pts)
            r = rid[sl][vr]
            out_f.append(f[ok]); out_r.append(r[ok])
            if with_t:
                out_t.append(mid[ok])
        if with_t:
            return torch.cat(out_f), torch.cat(out_r), torch.cat(out_t)
        return torch.cat(out_f), torch.cat(out_r)

    def _free_rays(self, org, dirs, d, s, cell=None, with_t: bool = False):
        if cell is None:
            cell = torch.full_like(d, True, dtype=torch.bool)
        t1 = torch.where(cell, d - (2 * s - self.v).clamp_min(0.0), d - 2 * s)
        t0 = torch.zeros_like(d)
        ft = None
        if with_t:
            ff, fr, ft = self._march(org, dirs, t0, t1, with_t=True)
        else:
            ff, fr = self._march(org, dirs, t0, t1)
        if len(ff):
            fe, oke = self.index(org + t1[:, None] * dirs)
            drop = oke[fr] & (ff == fe[fr])
            ff, fr = ff[~drop], fr[~drop]
            if ft is not None:
                ft = ft[~drop]
        return (ff, fr, ft) if with_t else (ff, fr)

    def _march_behind(self, org, dirs, d, s, step):
        if len(d) == 0:
            e = torch.zeros(0, dtype=torch.long, device=DEVICE)
            return e, e
        K = int(torch.ceil(s.max() / step).item())
        j = torch.arange(-K, K + 1, device=DEVICE, dtype=torch.float32) * step
        out_f, out_r = [], []
        per = max(1, CHUNK // (2 * K + 1))
        for a in range(0, len(d), per):
            sl = slice(a, a + per)
            t = d[sl, None] + j[None]
            val = (t - d[sl, None]).abs() <= s[sl, None] + 1e-9
            val = val & (j[None] >= 0)
            p = org[sl, None, :] + t[..., None] * dirs[sl, None, :]
            f, ok = self.index(p[val])
            r = torch.arange(a, a + len(d[sl]), device=DEVICE)[:, None].expand_as(t)[val]
            out_f.append(f[ok]); out_r.append(r[ok])
        return torch.cat(out_f), torch.cat(out_r)

    def camera_sigma(self, z: torch.Tensor, K) -> torch.Tensor:
        H, W = z.shape
        f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
        u = (torch.arange(W, device=z.device, dtype=z.dtype) + 0.5 - cx) / f
        vv = (torch.arange(H, device=z.device, dtype=z.dtype) + 0.5 - cy) / float(K[1, 1])
        P = torch.stack([u[None, :] * z, vv[:, None] * z, z], -1)
        s = int(self.p["normal_step"])
        theta = torch.zeros_like(z)
        c = P[s:H - s, s:W - s]
        du = P[s:H - s, 2 * s:] - P[s:H - s, :W - 2 * s]
        dv = P[2 * s:, s:W - s] - P[:H - 2 * s, s:W - s]
        n = torch.cross(du, dv, dim=-1)
        cosi = (n * c).sum(-1).abs() / (n.norm(dim=-1) * c.norm(dim=-1)).clamp_min(1e-12)
        zc, zl, zr, zu, zd = z[s:H - s, s:W - s], z[s:H - s, :W - 2 * s], z[s:H - s, 2 * s:], z[:H - 2 * s, s:W - s], z[2 * s:, s:W - s]
        okn = (zc > 0) & (zl > 0) & (zr > 0) & (zu > 0) & (zd > 0)
        for a_, b_ in ((zl, zr), (zu, zd)):
            okn = okn & ((b_ - 2 * zc + a_).abs() < 0.02 + 0.3 * torch.maximum((b_ - zc).abs(), (zc - a_).abs()))
        theta[s:H - s, s:W - s] = torch.where(okn, torch.arccos(cosi.clamp(0, 1)), torch.zeros_like(cosi))
        sig = depth_sigma(z.clamp(0.05, 5.0), theta)
        zn = torch.where(z > 0, z, torch.full_like(z, float("nan")))
        nb = torch.nn.functional.pad(zn[None, None], (1, 1, 1, 1), value=float("nan"))
        nb = torch.nn.functional.unfold(nb, 3)[0].T.view(H, W, 9)
        nb8 = torch.cat([nb[..., :4], nb[..., 5:]], -1)
        dev = torch.nan_to_num((nb8 - z[..., None]).abs(), nan=float("inf")).amin(-1)
        dev = torch.where(torch.isfinite(dev), dev, torch.zeros_like(dev))
        return torch.maximum(sig, dev)

    def _camera_delta(self, cam, depth, raw):
        base = self.p["base"]
        z = depth.reshape(-1); zr = raw.reshape(-1)
        sig = self.camera_sigma(depth, cam.K).reshape(-1)
        cosz = cam.cosz
        valid = z > 0
        seg = (zr > 0) & ~valid
        sig_seg = depth_sigma(zr.clamp(0.05, 5.0), torch.zeros_like(zr))
        d = torch.where(valid, z, zr) / cosz
        sr = torch.where(valid, sig, sig_seg) / cosz
        use = valid | seg
        idx = torch.nonzero(use).flatten()
        org = cam.pos.expand(len(idx), 3); dirs = cam.dirs[idx]
        self._robot_hit = None
        robot = torch.zeros_like(seg)
        if self._spheres is not None and bool(seg.any()):
            si = torch.nonzero(seg).flatten()
            ph = cam.pos + cam.dirs[si] * d[si, None]
            S = self._spheres
            gap = torch.cat([((ph[a:a + 8192, None] - S[None, :, :3]).norm(dim=-1) - S[None, :, 3]).amin(1) for a in range(0, len(ph), 8192)])
            hit = gap <= 2 * sr[si]
            robot[si[hit]] = True
            fh, okh = self.index(ph[hit])
            self._robot_hit = fh[okh]
        cell = (valid | robot)[idx]
        ff, fr, ft = self._free_rays(org, dirs, d[idx], sr[idx], cell, with_t=True)
        if bool(cell.any()) and self._spheres is not None and len(ff):
            near = cell[fr] & ~robot[idx][fr]
            thr = d[idx] - 2 * sr[idx] - self.v / 2
            near = near & (ft > thr[fr] - self.v)
            if bool(near.any()):
                ni = torch.nonzero(near).flatten()
                n1, n2 = self.n[1], self.n[2]
                f_ = ff[ni]
                cc = self.lo + (torch.stack([f_ // (n1 * n2), (f_ // n2) % n1, f_ % n2], -1).float() + 0.5) * self.v
                r_ = fr[ni]
                tc = ((cc - org[r_]) * dirs[r_]).sum(-1)
                close = tc > thr[r_]
                if bool(close.any()):
                    S = self._spheres
                    ci = torch.nonzero(close).flatten()
                    cc = cc[ci]
                    gap = torch.cat([((cc[a:a + 8192, None] - S[None, :, :3]).norm(dim=-1) - S[None, :, 3]).amin(1) for a in range(0, len(cc), 8192)])
                    bad = ci[gap < cam.seg_dist]
                    keep = torch.ones(len(ff), dtype=torch.bool, device=DEVICE); keep[ni[bad]] = False
                    ff = ff[keep]
        oi = torch.nonzero(valid & (sig <= self.p["cam_sig_max"])).flatten()
        of, orr = self._march_behind(cam.pos.expand(len(oi), 3), cam.dirs[oi], d[oi], sr[oi], self.v / 2)
        inc = base * torch.clamp(self.v / (2 * sig[oi]), max=1.0)
        return of, inc[orr], ff

    def _tof_delta(self, tof, out):
        base, alpha = self.p["base"], self.p["alpha"]
        n = int(self.p["tof_nsub"]); P = n * n
        rng = out["rng"]; mod = out["module"]
        fin = torch.isfinite(rng); noecho = torch.isinf(rng)
        zone_use = fin | noecho
        zi = torch.nonzero(zone_use).flatten()
        if len(zi) == 0:
            e = torch.zeros(0, dtype=torch.long, device=DEVICE)
            return e, e, torch.zeros(0, device=DEVICE), e, e
        dl = tof.sub_dirs_local(n)[out["zone"][zi]]
        dw = torch.einsum("zij,zpj->zpi", out["R"][zi], dl)
        d = torch.where(fin[zi], rng[zi], torch.full_like(rng[zi], TOF_RANGE))
        sig = torch.clamp(TOF_SIGMA_REL * d, min=TOF_SIGMA_FLOOR)
        org = out["org"][zi][:, None, :].expand(-1, P, 3).reshape(-1, 3)
        dirs = dw.reshape(-1, 3)
        dd = d[:, None].expand(-1, P).reshape(-1); ss = sig[:, None].expand(-1, P).reshape(-1)
        ff, fr = self._free_rays(org, dirs, dd, ss)
        f_mod = mod[zi][fr // P]
        hz = torch.nonzero(fin[zi]).flatten()
        if len(hz) == 0:
            e = torch.zeros(0, dtype=torch.long, device=DEVICE)
            return e, e, torch.zeros(0, device=DEVICE), f_mod, ff
        rays = (hz[:, None] * P + torch.arange(P, device=DEVICE)[None]).reshape(-1)
        o_org, o_dir = org[rays], dirs[rays]
        pc = o_org + dd[rays, None] * o_dir
        fc, okc = self.index(pc)
        zc = (rays // P)
        key = zc[okc] * self.N + fc[okc]
        uk = torch.unique(key)
        uk = uk[self.L.view(-1)[uk % self.N] >= -self.p["th_free"]]
        Nz = torch.zeros(len(zi), device=DEVICE).index_add_(0, uk // self.N, torch.ones(len(uk), device=DEVICE))
        of, orr = self._march_behind(o_org, o_dir, dd[rays], ss[rays], self.v / 2)
        zr = rays[orr] // P
        okz = Nz[zr] > 0
        okz = okz & (self.L.view(-1)[of] >= -self.p["th_free"])
        inc = base * torch.clamp(self.v / (2 * sig[zr]), max=1.0) / Nz[zr].clamp_min(1.0) ** alpha
        return mod[zi][zr][okz], of[okz], inc[okz], f_mod, ff

    @torch.no_grad()
    def integrate(self, cam, depth, raw, tof, tof_out, spheres):
        base, cap = self.p["base"], self.p["cap"]
        delta = torch.zeros(self.N, device=DEVICE)
        self._spheres = spheres[spheres[:, 3] > 0]
        if depth is not None:
            of, inc, ff = self._camera_delta(cam, depth, raw)
            om = self._occ_max; om.zero_(); om.scatter_reduce_(0, of, inc, reduce="amax")
            fm = self._free; fm.zero_(); fm[ff] = True
            delta += torch.where(om > 0, om, torch.where(fm, torch.full_like(om, -base), torch.zeros_like(om)))
        if tof_out is not None:
            om_mod, of, inc, fmod, ff = self._tof_delta(tof, tof_out)
            if len(of):
                ko = om_mod * self.N + of
                uo, inv = torch.unique(ko, return_inverse=True)
                vmax = torch.zeros(len(uo), device=DEVICE).scatter_reduce_(0, inv, inc, reduce="amax")
                delta.index_add_(0, uo % self.N, vmax)
            else:
                uo = torch.zeros(0, dtype=torch.long, device=DEVICE)
            if len(ff):
                uf = torch.unique(fmod * self.N + ff)
                uf = uf[~torch.isin(uf, uo)]
                delta.index_add_(0, uf % self.N, torch.full((len(uf),), -base, device=DEVICE))
        Lf = self.L.view(-1)
        Lf.add_(delta).clamp_(-cap, cap)
        if depth is not None and self._robot_hit is not None and len(self._robot_hit):
            f = self._robot_hit
            f = f[Lf[f] <= self.p["th_occ"]]
            Lf[f] = -cap
        self.set_free_spheres(spheres)

    def set_free_spheres(self, spheres: torch.Tensor):
        s = spheres[spheres[:, 3] > 0]
        if len(s) == 0:
            return
        lo = torch.floor(((s[:, :3] - s[:, 3:]).amin(0) - self.lo) / self.v).long().clamp_min(0)
        hi = torch.minimum(torch.floor(((s[:, :3] + s[:, 3:]).amax(0) - self.lo) / self.v).long() + 1, self.nt)
        if (hi <= lo).any():
            return
        blo = torch.maximum(torch.floor((s[:, :3] - s[:, 3:] - self.lo) / self.v).long() - 1, lo)
        bhi = torch.minimum(torch.floor((s[:, :3] + s[:, 3:] - self.lo) / self.v).long() + 2, hi)
        ext = (bhi - blo).clamp_min(0)
        B = [int(x) for x in ext.amax(0).tolist()]
        if min(B) <= 0:
            return
        off = torch.stack(torch.meshgrid(*[torch.arange(b, device=DEVICE) for b in B], indexing="ij"), -1).reshape(-1, 3)
        ijk = blo[:, None, :] + off[None]
        c = self.lo + (ijk.float() + 0.5) * self.v
        inside = ((c - s[:, None, :3]).pow(2).sum(-1) <= s[:, None, 3] ** 2) & (off[None] < ext[:, None, :]).all(-1)
        ijk = ijk[inside]
        f = (ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2]
        f = f[self.L.view(-1)[f] <= self.p["th_occ"]]
        self.L.view(-1)[f] = -self.p["cap"]

    def _pba(self, sites_mask: torch.Tensor, at: torch.Tensor = None) -> torch.Tensor:
        from curobo._src.perception.mapper.esdf.edt_parallel_banding import ParallelBandingEDT
        if self._edt is None:
            self._edt = ParallelBandingEDT(self.n, self.v, torch.device(DEVICE))
            ar = torch.arange(self.N, device=DEVICE, dtype=torch.int64)
            nz, ny = self.n[2], self.n[1]
            self._ijk = torch.stack([ar // (ny * nz), (ar // nz) % ny, ar % nz], -1).to(torch.int32)
            self._packed = (self._ijk[:, 0] | (self._ijk[:, 1] << 10) | (self._ijk[:, 2] << 20)).to(torch.int32)
        site = torch.where(sites_mask, self._packed, torch.full_like(self._packed, -1))
        if not bool(sites_mask.any()):
            return torch.full((self.N if at is None else len(at),), 1e4, device=DEVICE)
        self._edt.propagate(site)
        ijk = self._ijk
        if at is not None:
            site, ijk = site[at], ijk[at]
        ok = site >= 0
        dx = (site & 0x3FF) - ijk[:, 0]; dy = ((site >> 10) & 0x3FF) - ijk[:, 1]; dz = ((site >> 20) & 0x3FF) - ijk[:, 2]
        dist = (dx * dx + dy * dy + dz * dz).float().sqrt() * self.v
        return torch.where(ok, dist, torch.full_like(dist, 1e4))

    @torch.no_grad()
    def esdf(self) -> torch.Tensor:
        occ = (self.L > self.p["th_occ"]).view(-1)
        d_out = self._pba(occ)
        F = d_out - self.v / 2
        if bool(occ.any()):
            oi = torch.nonzero(occ).flatten()
            F[oi] = -(self._pba(~occ, at=oi) - self.v / 2)
        return F.view(self.n)

    @torch.no_grad()
    def distance_field(self, mask: torch.Tensor) -> torch.Tensor:
        F = self._pba(mask) - self.v / 2
        if bool(mask.any()):
            F = torch.where(mask, -(self._pba(~mask) - self.v / 2), F)
        return F.view(self.n)

    def cells_near(self, S: torch.Tensor, w: float, chunk: int = 128) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=DEVICE)
        if len(S) == 0:
            return m
        R = int(math.ceil((float(S[:, 3].max()) + w) / self.v)) + 1
        ar = torch.arange(-R, R + 1, device=DEVICE)
        off = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"), -1).reshape(-1, 3)
        for a in range(0, len(S), chunk):
            s = S[a:a + chunk]
            ijk = torch.floor((s[:, :3] - self.lo) / self.v).long()[:, None] + off[None]
            ctr = self.lo + (ijk.float() + 0.5) * self.v
            ok = ((ctr - s[:, None, :3]).norm(dim=-1) <= s[:, None, 3] + w) & ((ijk >= 0) & (ijk < self.nt)).all(-1)
            ijk = ijk[ok]
            m[(ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2]] = True
        return m
