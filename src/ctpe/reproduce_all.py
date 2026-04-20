from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


STAGES = [
    ("stage3", [sys.executable, "-m", "ctpe.reproduce_stage3"]),
    ("stage4", [sys.executable, "-m", "ctpe.reproduce_stage4_local"]),
    ("stage5", [sys.executable, "-m", "ctpe.reproduce_stage5"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main reproduction stages end to end.")
    parser.add_argument("--profile", choices=["smoke", "main", "heavy"], default="main")
    args = parser.parse_args()

    root = Path.cwd()
    (root / "logs").mkdir(exist_ok=True)

    for stage_name, cmd in STAGES:
        full = cmd + ["--profile", args.profile]
        start = time.time()
        print("Running:", " ".join(full), flush=True)
        proc = subprocess.run(full, cwd=root)
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise SystemExit(f"Stage {stage_name} failed after {elapsed:.1f}s")
        print(f"Completed {stage_name} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
