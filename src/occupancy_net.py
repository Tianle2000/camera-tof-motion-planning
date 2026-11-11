import torch
import torch.nn as nn

from config import DATA, DEVICE, MAP


def _block(ci, co):
    g = max(1, co // 4)
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(g, co), nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(g, co), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, ch, rates=(1, 2, 3)):
        super().__init__()
        g = max(1, ch // 4)
        self.br = nn.ModuleList([nn.Sequential(nn.Conv3d(ch, ch, 3, padding=r, dilation=r, bias=False), nn.GroupNorm(g, ch), nn.ReLU(inplace=True))
                                 for r in rates])
        self.fuse = nn.Sequential(nn.Conv3d(ch * len(rates), ch, 1, bias=False), nn.GroupNorm(g, ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.br], 1))


class OccupancyNet(nn.Module):
    def __init__(self, ch=(8, 16, 32), c0=4):
        super().__init__()
        c1, c2, c3 = ch
        self.e1, self.e2, self.e3 = _block(1, c1), _block(c1, c2), _block(c2, c3)
        self.pool = nn.MaxPool3d(2)
        self.mid = ASPP(c3)
        self.u3, self.d3 = nn.ConvTranspose3d(c3, c3, 2, stride=2), _block(2 * c3, c3)
        self.u2, self.d2 = nn.ConvTranspose3d(c3, c2, 2, stride=2), _block(2 * c2, c2)
        self.u1, self.d1 = nn.ConvTranspose3d(c2, c1, 2, stride=2), _block(2 * c1, c1)
        self.u0 = nn.ConvTranspose3d(c1, c0, 2, stride=2)
        self.d0 = nn.Sequential(nn.Conv3d(c0, c0, 3, padding=1, bias=False), nn.GroupNorm(1, c0), nn.ReLU(inplace=True))
        self.out = nn.Conv3d(c0, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        m = self.mid(self.pool(c))
        c = self.d3(torch.cat([self.u3(m), c], 1))
        b = self.d2(torch.cat([self.u2(c), b], 1))
        a = self.d1(torch.cat([self.u1(b), a], 1))
        return self.out(self.d0(self.u0(a)))


class OccupancyEstimator:
    def __init__(self):
        self.net = OccupancyNet().to(DEVICE)
        self.net.load_state_dict(torch.load(DATA / "occupancy_net.pt", map_location=DEVICE))
        self.net.eval()

    def predict(self, om) -> torch.Tensor:
        th = MAP["th_occ"]
        L1 = om.L
        nx, ny, nz = L1.shape
        Lc = L1.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(nx // 2, ny // 2, nz // 2, 8)
        mx = Lc.amax(-1)
        x2 = torch.zeros((112,) * 3, device=L1.device)
        x2[:nx // 2, :ny // 2, :nz // 2] = torch.where(mx > th, mx, Lc.mean(-1))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(self.net(x2[None, None].float()).float())[0, 0, :nx, :ny, :nz]
        return torch.where(L1 > th, torch.ones_like(p), torch.where(L1 < -th, torch.zeros_like(p), p))
