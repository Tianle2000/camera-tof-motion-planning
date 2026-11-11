# Manipulator motion planning in occluded and dynamic environments with a camera and proximity sensors

## Setup

Python 3.11 and an NVIDIA GPU.

```bash
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout 8e734f3ced1df898990bcd92de40abce475907db
pip install ".[cu13-torch]"
cd ..
pip install scipy
```

## Run

Static scenes:

```bash
python run.py --problems data/problems/static/hutch.pkl
```

Moving obstacles, with the moving-obstacle extension:

```bash
python run.py --problems data/problems/moving/crossing/hutch.pkl --extension
```

Static problems run at a control interval of 0.03 s. Problems with a moving obstacle run at 0.15 s, and camera and ToF frames enter the map one interval after they are taken. `--tof 16` uses 16 ToF sensors instead of 8. Results are written to `results/`, one problem per line.

## Data

| Path | Contents |
|---|---|
| `data/problems/static/` | 995 occluded static problems (hutch, cabinet, box, bookshelf, cubby) |
| `data/problems/mbm/` | 700 MotionBenchMaker problems |
| `data/problems/moving/` | 169 problems with a moving obstacle (crossing, approaching, held-out) |
| `data/occupancy_net.pt` | Occupancy estimation network |
| `data/tof_placement.json` | Camera-blind-spot placement of 16 ToF sensors; the first 8 form the 8-sensor placement |
| `data/tof_self_occlusion.npy` | Distance at which each ToF sub-ray hits the link it is mounted on |
| `data/camera.json` | Fixed camera pose |

Tested with torch 2.13 (CUDA 13.0), warp-lang 1.15.0 and numpy 2.4.6.
