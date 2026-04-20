from __future__ import annotations

import argparse
import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "ctpe.run_stage4_regime_refinement", "--profile", "{profile}"],
    [sys.executable, "-m", "ctpe.run_stage4_near_offpolicy_suite", "--profile", "{profile}"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-4 local extensions used by the manuscript.")
    parser.add_argument("--profile", choices=["smoke", "main", "heavy"], default="heavy")
    args = parser.parse_args()
    for cmd in COMMANDS:
        rendered = [part.format(profile=args.profile) for part in cmd]
        print(" ".join(rendered), flush=True)
        subprocess.run(rendered, check=True)


if __name__ == "__main__":
    main()
