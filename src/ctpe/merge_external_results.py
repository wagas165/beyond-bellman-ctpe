from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, UTC
from pathlib import Path


RECOGNIZED_CSV_PREFIXES = (
    "stage3_",
    "journal_",
    "stage2_",
)
RECOGNIZED_PLOT_PREFIXES = (
    "stage3_",
    "journal_",
)


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="External results directory or zip file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    figures = root / "figures"
    external = root / "external_results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    external.mkdir(parents=True, exist_ok=True)

    src = Path(args.source).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    extracted_dir: Path | None = None
    if src.is_file() and src.suffix.lower() == ".zip":
        tmpdir = Path(tempfile.mkdtemp(prefix="ctpe_ext_"))
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmpdir)
        extracted_dir = tmpdir
        copy_tree(tmpdir, external)
        scan_root = tmpdir
    elif src.is_dir():
        copy_tree(src, external)
        scan_root = src
    else:
        target = external / src.name
        shutil.copy2(src, target)
        scan_root = external

    copied_csv = 0
    copied_pdf = 0
    for path in scan_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv" and path.name.startswith(RECOGNIZED_CSV_PREFIXES):
            shutil.copy2(path, results / path.name)
            copied_csv += 1
        elif path.suffix.lower() == ".pdf" and path.name.startswith(RECOGNIZED_PLOT_PREFIXES):
            shutil.copy2(path, figures / path.name)
            copied_pdf += 1
        elif path.suffix.lower() == ".html" and path.name.startswith(RECOGNIZED_PLOT_PREFIXES):
            (figures / "interactive").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, figures / "interactive" / path.name)

    status_lines = [
        f"Source: {src}",
        f"Archived external copy: {external}",
        f"Copied CSV files into results/: {copied_csv}",
        f"Copied PDF files into figures/: {copied_pdf}",
    ]
    if copied_csv == 0:
        status_lines.append("Warning: no recognized CSV files were found, so numeric tables cannot be regenerated automatically.")
    status_path = root / "EXTERNAL_RESULTS_IMPORT_STATUS.txt"
    status_path.write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print(status_path)
    print("\n".join(status_lines))

    if extracted_dir is not None:
        pass


if __name__ == "__main__":
    main()
