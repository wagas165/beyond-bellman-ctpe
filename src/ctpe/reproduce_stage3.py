from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click stage-3 reproduction entry point.")
    parser.add_argument("--profile", choices=["smoke", "main", "heavy"], default="main")
    args = parser.parse_args()

    py = sys.executable
    run([py, "-m", "ctpe.run_stage3_heavy_benchmark", "--profile", args.profile])
    for section in ["dt,budget", "mismatch,feature", "startup,stability", "scaling"]:
        run([py, "-m", "ctpe.run_stage3_sensitivity_suite", "--profile", args.profile, "--sections", section])
    run([py, "-m", "ctpe.run_stage3_model_based_suite", "--profile", args.profile])


if __name__ == "__main__":
    main()
