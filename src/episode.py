import math
import time
from copy import deepcopy

import torch

import robot
from config import CAMERA_HZ, DEVICE, DT, EXECUTION, FLOOR, MOVING, MPC, PLANNER, REACH_TOLERANCE, RISK, TIME_LIMIT
from moving_obstacles import MovingObstacleDetector
from risk_cost import keep_mask
from scene import MovingBox, Scene, with_moving_box


def run(p: dict, rig, seed: int, extension: bool) -> dict:
    wall0 = time.time()
    ex, rk = EXECUTION, RISK
    obs_truth = with_moving_box(p) if p.get("moving") else p["obstacles"]
    truth = Scene(obs_truth)
    obs_render = deepcopy(obs_truth); obs_render.setdefault("cuboid", {})["__floor__"] = FLOOR
    render = Scene(obs_render)
    moving = MovingBox(p, obs_truth, obs_render, truth, render) if p.get("moving") else None
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    om, mp, mpc, risk = rig.map, rig.planner, rig.mpc, rig.risk
    rig.tof.reset()
    om.reset()
    mpc.reset_seed(); mp.reset_seed()
    q = torch.as_tensor(p["start"], dtype=torch.float32, device=DEVICE)
    goal_xyz = torch.as_tensor(p["goal_pose"]["position_xyz"], dtype=torch.float32, device=DEVICE)
    lo, vox = rig.esdf_lo, om.v
    log = dict(n_replans=0, n_stuck=0, n_hold=0)
    risk.set_weight(0.0)
    detector = MovingObstacleDetector(rig.camera) if extension else None
    d_moving = []
    hit_while_waiting = False
    P = None
    risk_state = dict(n_replans=0)
    t, step, cam_last = 0.0, 0, 0
    n_ctl = ex["moving_control_steps"] if moving is not None else 1
    lag = n_ctl * DT if n_ctl > 1 else 0.0
    queue = []

    def sense(t_now, q_now, spheres_now, cam_due):
        depth = raw = tof_out = None
        if cam_due:
            depth, raw = rig.camera.render(render, spheres_now, gen)
        due = rig.tof.due(t_now)
        if bool(due.any()):
            tof_out = rig.tof.read(q_now, render, due, gen)
        om.integrate(rig.camera, depth, raw, rig.tof, tof_out, spheres_now)
        if detector is not None and depth is not None:
            detector.camera_frame(t_now, depth)

    def sense_delayed(t_now, q_now, spheres_now, cam_due):
        depth = raw = tof_out = None
        if cam_due:
            depth, raw = rig.camera.render(render, spheres_now, gen)
        due = rig.tof.due(t_now)
        if bool(due.any()):
            tof_out = rig.tof.read(q_now, render, due, gen)
        queue.append((t_now, spheres_now.clone(), depth, raw, tof_out))
        while queue and queue[0][0] + lag <= t_now + 1e-9:
            t_f, s_f, d_f, r_f, tof_f = queue.pop(0)
            om.integrate(rig.camera, d_f, r_f, rig.tof, tof_f, s_f)
            if detector is not None and d_f is not None:
                detector.camera_frame(t_f, d_f)

    def fields():
        return om.esdf(), om.distance_field((om.L > ex["obstacle_threshold"]).view(-1))

    def with_moving(F):
        if detector is None or not detector.active(t):
            return F
        m = detector.swept_mask(om, t)
        if m is None:
            return F
        return torch.minimum(F, om.distance_field(m) - (MOVING["activation_distance"] - MPC["activation_distance"]))

    def margin(F, s):
        return float((robot.esdf_at(F, lo, vox, s[:, :3]) - s[:, 3]).min())

    def predict():
        nonlocal P
        P = rig.estimator.predict(om).reshape(-1).float()
        Lf = om.L.view(-1)
        part = (Lf > rk["partial_evidence"]) & (Lf <= om.p["th_occ"])
        P = torch.where(part, P.clamp_min(rk["partial_probability"]), P)

    def plan_with_risk(q_now):
        predict()
        s_now, _ = rig.spheres(q_now.view(1, -1)); s_now = s_now[0][s_now[0][:, 3] > 0]
        risk.load(om, P, keep_mask(om, goal_xyz, s_now))
        w = rk["weight"]
        risk.set_weight(w)
        out = robot.plan(mp, q_now, p, rk["attempts"])
        if out is not None:
            r0 = risk.route(rig, out[0], out[1])
            if r0 > rk["escalate_threshold"]:
                risk.set_weight(w * rk["escalate"])
                alt = robot.plan(mp, q_now, p, rk["attempts"])
                risk.set_weight(w)
                if alt is not None and risk.route(rig, alt[0], alt[1]) < r0:
                    out = alt
        if out is None:
            risk.set_weight(w * rk["fallback_fraction"])
            out = robot.plan(mp, q_now, p, PLANNER["max_attempts"])
        if out is None:
            risk.set_weight(0.0)
            out = robot.plan(mp, q_now, p, PLANNER["max_attempts"])
        risk.set_weight(0.0)
        if out is not None:
            risk_state.update(planned=risk.route(rig, out[0], out[1]), step=step)
        return out

    def plan_or_wait(q_now):
        nonlocal t, step, cam_last, Fp, Fm, hit_while_waiting
        rig.load_world(Fp)
        out = plan_with_risk(q_now)
        if out is not None:
            return out, False
        t_w0 = t
        sph0, _ = rig.spheres(q_now.view(1, -1)); sph0 = sph0[0].clone()
        s0 = sph0[sph0[:, 3] > 0]

        def keep_waiting():
            if t >= TIME_LIMIT:
                return False
            if t - t_w0 < ex["retry_s"] - 1e-9:
                return True
            return detector is not None and detector.has_targets() and t - t_w0 < MOVING["wait_s"] - 1e-9

        n_wait = ex["retry_every"] if n_ctl == 1 else n_ctl * math.ceil(ex["retry_every"] / n_ctl)
        while out is None and keep_waiting():
            for _ in range(n_wait):
                t += DT; step += 1
                if moving is not None:
                    moving.at(t)
                    dm = moving.sdf(s0[:, :3]) - s0[:, 3]
                    d_moving.append(float(dm.min()))
                    if bool((dm < 0).any()):
                        hit_while_waiting = True
                        log.update(collision_depth_m=float(-dm.min()), collision_moving=True, collision_while_waiting=True)
                        return None, True
                cam_frame = int(math.floor(t * CAMERA_HZ))
                if n_ctl == 1:
                    sense(t, q_now, sph0, cam_frame > cam_last); cam_last = max(cam_last, cam_frame)
                elif step % n_ctl == 0:
                    sense_delayed(t, q_now, sph0, cam_frame > cam_last); cam_last = max(cam_last, cam_frame)
            Fp, Fm = fields()
            rig.load_world(Fp)
            out = plan_with_risk(q_now)
        return out, True

    def steps(traj, sec):
        return max(1, int(round(sec / traj[1])))

    def reference(traj, idx):
        return robot.sample(traj, min(idx * traj[1] + ex["progress_lead"], (len(traj[0]) - 1) * traj[1]))

    sph, ee = rig.spheres(q.view(1, -1)); ee = ee.clone()
    s = sph[0][sph[0][:, 3] > 0]
    if detector is not None:
        detector.prewarm(lambda g: rig.camera.render(render, sph[0], g)[0], seed)
    sense(0.0, q, sph[0], True)
    Fp, Fm = fields()
    Fw, Fw_src = with_moving(Fm), Fm
    traj, _ = plan_or_wait(q)
    outcome = None
    path_len = 0.0
    if traj is None:
        outcome = "collision" if hit_while_waiting else "no_solution"
    else:
        if Fw_src is not Fm:
            Fw, Fw_src = with_moving(Fm), Fm
        rig.load_world(Fw)
        state = robot.mpc_start(mpc, q)
        idx, win_n = 0, steps(traj, ex["window"])
        robot.mpc_set_goal(mpc, reference(traj, idx))
        m_cur = margin(Fw, s)
        prog_idx, prog_t, reach_t0 = 0, 0.0, None
        zero = torch.zeros(7, device=DEVICE)
    goal_zone = robot.MpcGoalZone(mpc)
    while outcome is None:
        goal_zone.update(ee[0], goal_xyz)
        for sub in range(n_ctl):
            if sub > 0:
                robot.mpc_set_goal(mpc, robot.sample(traj, min(idx * traj[1] + ex["progress_lead"] + sub * DT, (len(traj[0]) - 1) * traj[1])))
            nxt = robot.mpc_step(mpc, state)
            if nxt is None:
                outcome = "solver_nan"; break
            qn, qdn, qddn = nxt
            sn, _ = rig.spheres(qn.view(1, -1)); sn = sn[0][sn[0][:, 3] > 0]
            m_new = margin(Fw, sn)
            held = m_new < 0 and m_new < m_cur - 1e-4
            if held:
                log["n_hold"] += 1
                qn, qdn, qddn = q.clone(), zero, zero
            t += DT; step += 1
            if moving is not None:
                moving.at(t)
            q = qn
            state = robot.mpc_state(mpc, q, qdn, qddn)
            if held:
                mpc.reset_robot(state)
            sph, ee_new = rig.spheres(q.view(1, -1))
            path_len += float((ee_new - ee).norm()); ee = ee_new.clone()
            s = sph[0][sph[0][:, 3] > 0]
            pen = truth.sdf(s[:, :3]) - s[:, 3]
            if moving is not None:
                d_moving.append(float((moving.sdf(s[:, :3]) - s[:, 3]).min()))
            if bool((pen < 0).any()):
                log["collision_depth_m"] = float(-pen.min())
                if moving is not None:
                    log["collision_moving"] = bool(((moving.sdf(s[:, :3]) - s[:, 3]) < 0).any())
                outcome = "collision"; break
            if sub < n_ctl - 1:
                m_cur = margin(Fw, s)
        if outcome is not None:
            break
        cam_frame = int(math.floor(t * CAMERA_HZ))
        (sense if n_ctl == 1 else sense_delayed)(t, q, sph[0], cam_frame > cam_last); cam_last = max(cam_last, cam_frame)
        Fp, Fm = fields()
        Fw, Fw_src = with_moving(Fm), Fm
        m_cur = margin(Fw, s)
        qs = traj[0]; T = len(qs)
        dq = (qs[idx:min(T, idx + win_n + 1)] - q).norm(dim=-1)
        idx = idx + min(int(torch.argmin(dq)), ex["progress_max_advance"] * n_ctl)
        at_end = idx >= T - 2 and float((q - qs[-1]).norm()) < ex["end_q"]
        err = float((ee[0] - goal_xyz).norm())
        settled = float(qdn.norm()) < ex["settle_velocity"]
        if settled and err < REACH_TOLERANCE:
            log["final_pos_err_m"] = err
            outcome = "success"; break
        if at_end and settled:
            reach_t0 = t if reach_t0 is None else reach_t0
            if t - reach_t0 >= ex["reach_wait"]:
                log["final_pos_err_m"] = err
                outcome = "reach_error"; break
        else:
            reach_t0 = None
        if t >= TIME_LIMIT:
            outcome = "timeout"; break
        rest = qs[idx:]
        rest = rest[::max(1, len(rest) // 20)]
        rs, _ = rig.spheres(rest)
        rs = rs.reshape(-1, 4); rs = rs[rs[:, 3] > 0]
        blocked = bool(((robot.esdf_at(Fp, lo, vox, rs[:, :3]) - rs[:, 3]) < 0).any())
        if not blocked:
            blocked = bool(((robot.esdf_at(Fm, lo, vox, rs[:, :3]) - rs[:, 3]) < 0).any())
        if idx > prog_idx:
            prog_idx, prog_t = idx, t
        stuck = (not at_end) and t - prog_t >= ex["stuck_s"]
        risky = False
        if (not blocked and not stuck and risk_state["n_replans"] < rk["replan_max"]
                and step - risk_state.get("step", 0) >= rk["predict_every"] and len(qs) - idx >= 4):
            risk_state["step"] = step
            predict()
            risk.load(om, P, keep_mask(om, goal_xyz, s))
            if risk.route(rig, qs[idx:], traj[1]) > max(rk["replan_threshold"], rk["replan_ratio"] * risk_state.get("planned", 0.0)):
                risky = True; risk_state["n_replans"] += 1
        if blocked or stuck or risky:
            log["n_replans" if (blocked or risky) else "n_stuck"] += 1
            traj, waited = plan_or_wait(q)
            if traj is None:
                outcome = "collision" if hit_while_waiting else "no_solution"; break
            if waited:
                state = robot.mpc_state(mpc, q); mpc.reset_robot(state)
            idx, win_n = 0, steps(traj, ex["window"])
            prog_idx, prog_t = 0, t
        if Fw_src is not Fm:
            Fw, Fw_src = with_moving(Fm), Fm
        rig.load_world(Fw)
        robot.mpc_set_goal(mpc, reference(traj, idx))
    goal_zone.restore()
    risk.set_weight(0.0)
    log.update(outcome=outcome, t_world_s=t, path_len_m=path_len, wall_s=time.time() - wall0)
    if moving is not None:
        log["min_distance_to_moving_m"] = round(min(d_moving), 4) if d_moving else None
    return log
