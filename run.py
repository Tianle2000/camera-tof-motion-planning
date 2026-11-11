import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import episode
import robot
from config import START_INSET
from rig import Rig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--tof", type=int, default=8, choices=[8, 16])
    ap.add_argument("--extension", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    groups = pickle.loads(Path(a.problems).read_bytes())
    name = f"{Path(a.problems).parent.name}_{Path(a.problems).stem}_tof{a.tof}" + ("_extension" if a.extension else "")
    out = Path(a.out) if a.out else Path("results") / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add((r["group"], r["index"]))
    t0 = time.time()
    rig = Rig(a.tof)
    print(f"setup {time.time() - t0:.0f} s", flush=True)
    with open(out, "a") as f:
        for g in sorted(groups):
            indices = sorted(groups[g])
            for i in indices[:a.first] if a.first else indices:
                if (g, i) in done:
                    continue
                p = robot.inset_start(groups[g][i], rig.kin, START_INSET)
                r = episode.run(p, rig, seed=i, extension=a.extension)
                r = dict(group=g, index=int(i), **r)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(f"{g}[{i}] {r['outcome']:12s} {r['t_world_s']:.2f} s", flush=True)
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
