from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = [
    [sys.executable, "-m", "ctpe.run_stage5_clean_regime_suite"],
    [sys.executable, "-m", "ctpe.run_stage5_operating_regime_diagnostic"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click stage-5 reproduction entry point.")
    parser.add_argument("--profile", choices=["smoke", "main", "heavy"], default="heavy")
    args = parser.parse_args()

    root = Path.cwd()
    (root / "logs").mkdir(exist_ok=True)

    for cmd in COMMANDS:
        full = cmd + ["--profile", args.profile]
        log_name = Path(cmd[-1]).stem + f"_{args.profile}.log"
        log_path = root / "logs" / log_name
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(full, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            f.write(proc.stdout)
        if proc.returncode != 0:
            raise SystemExit(f"Command failed: {' '.join(full)}\nSee {log_path}")


if __name__ == "__main__":
    main()
