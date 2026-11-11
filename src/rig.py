import json

import numpy as np
import torch

import robot
from config import DATA, DEVICE, GRID_CENTER, GRID_EXTENT, TRUNCATION, VOXEL
from occupancy_map import OccupancyMap
from occupancy_net import OccupancyEstimator
from risk_cost import RiskCost, install
from sensors import Camera, ToFArray


def voxel_template(n_tof: int):
    from curobo.perception import Mapper, MapperCfg
    cfg = MapperCfg(voxel_size=VOXEL, extent_meters_xyz=GRID_EXTENT, grid_center=torch.tensor(GRID_CENTER),
                    truncation_distance=TRUNCATION, seeding_method="gather",
                    depth_maximum_distance=5.0, depth_minimum_distance=0.02 if n_tof else 0.05,
                    minimum_tsdf_weight=0.3, decay_factor=0.90, frustum_decay_factor=0.85,
                    enable_static=False, num_cameras=1 + n_tof, image_height=120, image_width=160, block_size=8,
                    esdf_voxel_size=VOXEL, extent_esdf_meters_xyz=GRID_EXTENT)
    return Mapper(cfg).compute_esdf()


class Rig:
    def __init__(self, n_tof: int):
        cam = json.loads((DATA / "camera.json").read_text())
        self.camera = Camera(cam["position"], cam["rotation"])
        placement = json.loads((DATA / "tof_placement.json").read_text())[:n_tof]
        mounts = [(m["link"], np.array(m["position"]), np.array(m["rotation"])) for m in placement]
        kin_tof = robot.kinematics(sorted({m[0] for m in mounts}))
        self.tof = ToFArray(mounts, kin_tof, np.load(DATA / "tof_self_occlusion.npy")[:n_tof], seed=0)
        self.map = OccupancyMap(GRID_CENTER, GRID_EXTENT, VOXEL)
        vg = voxel_template(n_tof)
        nx, ny, nz = (round(x / vg.voxel_size) for x in vg.dims)
        self.esdf_shape = (nx, ny, nz)
        self.esdf_lo = torch.tensor(vg.pose[:3], device=DEVICE) - torch.tensor([nx, ny, nz], device=DEVICE) * vg.voxel_size / 2
        assert tuple(int(v) for v in self.map.n) == self.esdf_shape
        self.risk = None

        def add_risk_cost(mp):
            self.risk = RiskCost(self.map)
            install(mp.trajopt_solver, self.risk)

        from curobo._src.geom.types import SceneCfg
        self.planner = robot.make_planner(SceneCfg(voxel=[vg]), {"voxel": {"layers": 1, "dims": list(GRID_EXTENT), "voxel_size": VOXEL}}, add_risk_cost)
        self.vox_buf = self.planner.scene_collision_checker.data.voxels.features[0, 0, :nx * ny * nz, 0]
        self.kin = robot.kinematics(("panda_hand",))
        self.mpc = robot.make_mpc(self)
        self.estimator = OccupancyEstimator()

    def spheres(self, q: torch.Tensor):
        from curobo._src.state.state_joint import JointState
        q = q.reshape(-1, 7).contiguous()
        ks = self.kin.compute_kinematics(JointState.from_position(q))
        ee = ks.tool_poses.position.view(len(q), -1, 3)[:, 0]
        return ks.robot_spheres.view(len(q), -1, 4), ee

    def load_world(self, F: torch.Tensor):
        self.vox_buf.copy_(F.reshape(-1).to(self.vox_buf.dtype))
