from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


OVERFULL_PAT = re.compile(r'Overfull \\hbox|Overfull \\vbox')


def resolve_bibliography_engine() -> list[str]:
    for candidate in ('bibtex', 'bibtex.original', 'bibtex8', 'bibtexu'):
        path = shutil.which(candidate)
        if path:
            return [path]
    raise FileNotFoundError('No bibtex-compatible executable found on PATH.')


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def compile_tex(stem: str, cwd: Path) -> list[str]:
    run(['pdflatex', '-interaction=nonstopmode', f'{stem}.tex'], cwd)
    run(resolve_bibliography_engine() + [stem], cwd)
    run(['pdflatex', '-interaction=nonstopmode', f'{stem}.tex'], cwd)
    run(['pdflatex', '-interaction=nonstopmode', f'{stem}.tex'], cwd)
    log_path = cwd / f'{stem}.log'
    warnings: list[str] = []
    if log_path.exists():
        for line in log_path.read_text(errors='ignore').splitlines():
            if OVERFULL_PAT.search(line):
                warnings.append(line.strip())
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description='Compile LaTeX files and report overfull-box warnings.')
    parser.add_argument('--strict', action='store_true', help='Exit with an error if overfull boxes are found.')
    parser.add_argument('--targets', nargs='*', default=['main', 'main_nonanonymous', 'supplement'])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    all_warnings: list[tuple[str, str]] = []
    for stem in args.targets:
        warnings = compile_tex(stem, root)
        if warnings:
            for w in warnings:
                all_warnings.append((stem, w))
        print(f'[compiled] {stem}.pdf')

    if all_warnings:
        print('\nOverfull warnings found:')
        for stem, warning in all_warnings:
            print(f'  [{stem}] {warning}')
        if args.strict:
            raise SystemExit(1)
    else:
        print('\nNo overfull hbox/vbox warnings found in the compiled targets.')


if __name__ == '__main__':
    main()
