import math
import types
from copy import deepcopy

import numpy as np
import torch

from config import DEVICE, DT, MPC, PLANNER


def robot_cfg_dict(tool_frames=("panda_hand",)) -> dict:
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml")); cfg = cfg.get("robot_cfg", cfg)
    k = cfg["kinematics"]
    k["collision_sphere_buffer"] = 0.0
    if "attached_object" in k["collision_link_names"]:
        k["collision_link_names"].remove("attached_object")
    k["tool_frames"] = list(tool_frames)
    k["lock_joints"] = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
    cfg["load_dynamics"] = False
    return cfg


def kinematics(tool_frames):
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.robot import RobotCfg
    return Kinematics(RobotCfg.create(deepcopy(robot_cfg_dict(tuple(tool_frames))), device_cfg=DeviceCfg()).kinematics)


def collision_activation_tensors(solver) -> list:
    out = {}
    for s in (solver, getattr(solver, "core", None)):
        for r in (getattr(s, "optimizer_rollouts", None) or []):
            for mn in ("cost_manager", "constraint_manager", "hybrid_cost_constraint_manager"):
                sc = getattr(getattr(getattr(r, mn, None), "config", None), "scene_collision_cfg", None)
                t = getattr(sc, "activation_distance", None)
                if torch.is_tensor(t):
                    out[t.data_ptr()] = t
    return list(out.values())


def _plan_pose_single(self, goal_tool_poses, current_state, max_attempts: int, enable_graph_attempt: int):
    trajopt_result = None
    total_time = 0.0
    solve_time = 0.0
    og_current_state = current_state.clone()
    num_seeds = self.trajopt_solver.config.num_seeds
    for current_attempt in range(max_attempts):
        current_state = og_current_state.clone()
        ik_result = self.ik_solver.solve_pose(goal_tool_poses, return_seeds=num_seeds, current_state=current_state)
        total_time += ik_result.total_time
        solve_time += ik_result.solve_time
        success_count = torch.count_nonzero(ik_result.success)
        if success_count == 0:
            continue
        seed_config = ik_result.solution
        if success_count < num_seeds:
            good_solution = seed_config[ik_result.success][0:1, :].clone()
            seed_config[~ik_result.success] = good_solution
        seed_traj = None
        finetune_attempts = 1
        finetune_dt_scale = 0.55
        if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
            graph_seed = self._get_graph_seed_trajectories(current_state, seed_config)
            if graph_seed is None:
                continue
            seed_traj = graph_seed
            finetune_attempts = 3
            finetune_dt_scale = 0.75
        trajopt_result = self.trajopt_solver.solve_pose(goal_tool_poses, current_state, seed_config=seed_config, seed_traj=seed_traj,
                                                        use_implicit_goal=True, finetune_attempts=finetune_attempts, finetune_dt_scale=finetune_dt_scale)
        total_time += trajopt_result.total_time
        solve_time += trajopt_result.solve_time
        if torch.count_nonzero(trajopt_result.success) > 0:
            break
    if trajopt_result is not None:
        trajopt_result.total_time = total_time
        trajopt_result.solve_time = solve_time
    return trajopt_result


def make_planner(scene_cfg, collision_cache, cost_hook):
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.robot import RobotCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    robot = RobotCfg.create(deepcopy(robot_cfg_dict()), device_cfg=DeviceCfg())
    cfg = MotionPlannerCfg.create(robot=robot, scene_model=scene_cfg, use_cuda_graph=True, num_ik_seeds=32, num_trajopt_seeds=4,
                                  optimizer_collision_activation_distance=PLANNER["activation_distance"], store_debug=False,
                                  collision_cache=collision_cache)
    mp = MotionPlanner(cfg)
    mp._plan_pose_single = types.MethodType(_plan_pose_single, mp)
    cost_hook(mp)
    mp.warmup(enable_graph=True)
    mp.graph_planner.config.new_nodes_per_iteration = int(PLANNER["graph_nodes"])
    return mp


def _goal(mp, p):
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose
    return GoalToolPose.from_poses({mp.tool_frames[0]: Pose.from_list(p["goal_pose"]["position_xyz"] + p["goal_pose"]["quaternion_wxyz"])},
                                   ordered_tool_frames=mp.tool_frames)


def _plan_once(mp, q: torch.Tensor, p: dict, max_attempts: int):
    from curobo._src.state.state_joint import JointState
    mp.reset_seed()
    res = mp.plan_pose(_goal(mp, p), JointState.from_position(q.view(1, -1)), max_attempts=max_attempts, enable_graph_attempt=1)
    if res is None or not res.success.item():
        return None
    traj = res.get_interpolated_plan()
    if traj.joint_names is not None and list(traj.joint_names) != list(mp.joint_names):
        traj = traj.reorder(mp.joint_names)
    qs = traj.position.view(-1, traj.position.shape[-1])[:, :7].clone()
    dt = float(traj.dt.flatten()[0].item()) if torch.is_tensor(traj.dt) else float(traj.dt)
    return qs, dt


def plan(mp, q: torch.Tensor, p: dict, max_attempts: int):
    out = _plan_once(mp, q, p, max_attempts)
    if out is None:
        ts = collision_activation_tensors(mp.ik_solver) + collision_activation_tensors(mp.trajopt_solver)
        old = [t.clone() for t in ts]
        try:
            for t in ts:
                t[:] = PLANNER["fallback_activation_distance"]
            out = _plan_once(mp, q, p, max_attempts)
        finally:
            for t, o in zip(ts, old):
                t.copy_(o)
    return out


def sample(traj, tau: float) -> torch.Tensor:
    qs, dt = traj
    x = tau / dt
    i = int(math.floor(x))
    if i >= len(qs) - 1:
        return qs[-1]
    a = x - i
    return qs[i] * (1 - a) + qs[i + 1] * a


def make_mpc(rig):
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util.config_io import load_yaml
    from curobo.content import get_task_configs_path
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    task = deepcopy(load_yaml(str(get_task_configs_path() / "mpc" / "lbfgs_mpc.yml")))
    task["rollout"]["cost_cfg"]["cspace_cfg"]["activation_distance"][0] = MPC["joint_limit_activation_distance"]
    robot = RobotCfg.create(deepcopy(robot_cfg_dict()), device_cfg=DeviceCfg())
    cfg = ModelPredictiveControlCfg.create(robot=robot, optimizer_configs=[task], scene_model=None, use_cuda_graph=True,
                                           optimization_dt=DT, interpolation_steps=4,
                                           optimizer_collision_activation_distance=MPC["activation_distance"])
    mpc = ModelPredictiveControl(cfg, scene_collision_checker=rig.planner.scene_collision_checker)
    jl = robot.kinematics.kinematics_config.joint_limits.position
    mpc._joint_lo, mpc._joint_hi = jl[0][:7].to(DEVICE).float(), jl[1][:7].to(DEVICE).float()
    return mpc


def mpc_state(mpc, q: torch.Tensor, qd: torch.Tensor = None, qdd: torch.Tensor = None):
    from curobo._src.state.state_joint import JointState
    q = q.reshape(1, -1).clone()
    s = JointState.from_position(q, joint_names=mpc.joint_names)
    s.velocity = torch.zeros_like(q) if qd is None else qd.reshape(1, -1).clone()
    s.acceleration = torch.zeros_like(q) if qdd is None else qdd.reshape(1, -1).clone()
    return s


def mpc_start(mpc, q: torch.Tensor):
    mpc.setup(mpc_state(mpc, q))
    mpc.disable_tool_pose_tracking()
    mpc.enable_joint_position_tracking()
    return mpc_state(mpc, q)


def mpc_set_goal(mpc, q_goal: torch.Tensor):
    from curobo._src.state.state_joint import JointState
    mpc.update_goal_state(JointState.from_position(q_goal.reshape(1, -1).clone(), joint_names=mpc.joint_names))


def mpc_step(mpc, state):
    r = mpc.optimize_action_sequence(state)
    seq = r.action_sequence
    if seq is None or seq.position.shape[1] == 0:
        return None
    q, qd, qdd = seq.position[0, -1, :7], seq.velocity[0, -1, :7], seq.acceleration[0, -1, :7]
    if not (torch.isfinite(q).all() and torch.isfinite(qd).all()):
        return None
    return torch.clamp(q, mpc._joint_lo, mpc._joint_hi), qd.clone(), qdd.clone()


class MpcGoalZone:
    def __init__(self, mpc):
        self.ts = collision_activation_tensors(mpc)
        self.base = [t.clone() for t in self.ts]
        self.inside = False

    def update(self, ee_xyz, goal_xyz):
        inside = float((ee_xyz - goal_xyz).norm()) < MPC["goal_zone"]
        if inside != self.inside:
            if inside:
                for t in self.ts:
                    t[:] = float(MPC["goal_zone_activation_distance"])
            else:
                self.restore()
            self.inside = inside

    def restore(self):
        for t, b in zip(self.ts, self.base):
            t.copy_(b)
        self.inside = False


_SHAPE_T = {}
_CORNERS = {}


def esdf_at(F: torch.Tensor, lo: torch.Tensor, vs: float, pts: torch.Tensor) -> torch.Tensor:
    key = (tuple(F.shape), str(pts.device))
    n = _SHAPE_T.get(key)
    if n is None:
        n = _SHAPE_T[key] = torch.tensor(F.shape, device=pts.device)
    v = (pts - lo) / vs - 0.5
    i0 = torch.floor(v).long(); f = v - i0.float()
    inside = ((i0 >= 0) & (i0 + 1 < n)).all(-1)
    i0 = torch.minimum(i0.clamp_min(0), n - 2)
    s = torch.zeros(len(pts), device=pts.device)
    Ff = F.float().reshape(-1)
    ny, nz = int(F.shape[1]), int(F.shape[2])
    base = (i0[:, 0] * ny + i0[:, 1]) * nz + i0[:, 2]
    key = (ny, nz, str(pts.device))
    if key not in _CORNERS:
        c = torch.tensor([(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], device=pts.device)
        _CORNERS[key] = ((c[:, 0] * ny + c[:, 1]) * nz + c[:, 2], c.bool())
    off, m = _CORNERS[key]
    g = 1 - f
    W = torch.where(m[None, :, 0], f[:, None, 0], g[:, None, 0]) * torch.where(m[None, :, 1], f[:, None, 1], g[:, None, 1]) \
        * torch.where(m[None, :, 2], f[:, None, 2], g[:, None, 2])
    P = Ff[base[:, None] + off[None]] * W
    for j in range(8):
        s = s + P[:, j]
    return torch.where(inside, s, torch.full_like(s, 10.0))


def inset_start(p: dict, kin, inset: float):
    jl = kin.kinematics_config.joint_limits.position
    lo, hi = jl[0][:7].cpu().numpy().astype(float), jl[1][:7].cpu().numpy().astype(float)
    q = np.array(p["start"], dtype=float)
    q7 = np.clip(q[:7], lo + inset, hi - inset)
    if float(np.abs(q7 - q[:7]).max()) == 0.0:
        return p
    q[:7] = q7
    return {**p, "start": q.tolist()}
