from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DEVICE = "cuda"

GRID_CENTER = (0.0, 0.0, 0.45)
GRID_EXTENT = (2.1, 2.2, 2.0)
VOXEL = 0.01
TRUNCATION = 0.04

DT = 0.03
CAMERA_HZ = 30.0
TIME_LIMIT = 20.0
REACH_TOLERANCE = 0.01
START_INSET = 0.01
FLOOR = {"dims": [20.0, 20.0, 0.1], "pose": [0.0, 0.0, -0.85, 1.0, 0.0, 0.0, 0.0]}

TOF_RANGE = 0.3

MAP = dict(base=0.5, cap=1.0, th_occ=0.2, th_free=0.2, alpha=1.0, cam_sig_max=0.03, tof_nsub=4, normal_step=2)

PLANNER = dict(activation_distance=0.07, fallback_activation_distance=0.0025, graph_nodes=100, max_attempts=20)
MPC = dict(activation_distance=0.01, joint_limit_activation_distance=0.001,
           goal_zone=0.08, goal_zone_activation_distance=0.0025)

EXECUTION = dict(obstacle_threshold=0.05, window=0.5, stuck_s=1.0, settle_velocity=0.1, reach_wait=1.0, end_q=0.1,
                 retry_s=0.2, retry_every=7, progress_lead=0.15, progress_max_advance=3, moving_control_steps=5)

RISK = dict(weight=1e4, attempts=4, escalate=5.0, escalate_threshold=0.2, fallback_fraction=0.1,
            replan_threshold=0.05, replan_ratio=1.5, replan_max=6, predict_every=10,
            partial_evidence=0.05, partial_probability=0.9,
            eta=0.03, thresholds=(0.15, 0.4, 0.7), probability_floor=0.05, goal_zone=0.08, arm_band=0.02, stride=2)

MOVING = dict(k_sig=4.0, min_diff=0.02, n_warm=5, alpha=0.1, absorb_s=1.0, n_min=8, gate=0.10, conf_hits=3, v_min=0.05,
              keep_s=0.4, hist=6, horizon=0.5, h_step=0.05, extrude=0.05, activation_distance=0.05, max_pts=400, wait_s=3.0,
              prewarm=30)
