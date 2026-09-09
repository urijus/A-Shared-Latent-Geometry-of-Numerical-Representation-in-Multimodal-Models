"""Run fast, CPU-only checks for the publication-facing repository surface."""

from __future__ import annotations

import compileall
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src", "experiments", "visualizations")
ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".feather",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".jsonl",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tsv",
    ".wav",
    ".webp",
}


def tracked_files() -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in process.stdout.splitlines() if line]


def public_entrypoints() -> list[Path]:
    patterns = ("experiments/0*/*.py", "visualizations/0*/*.py")
    return sorted(
        path
        for pattern in patterns
        for path in ROOT.glob(pattern)
        if path.name != "__init__.py"
    )


def main() -> int:
    for source_root in SOURCE_ROOTS:
        if not compileall.compile_dir(ROOT / source_root, quiet=1):
            return 1

    tracked = [path for path in tracked_files() if path.exists()]
    artifacts = [path for path in tracked if path.suffix.lower() in ARTIFACT_SUFFIXES]
    shell_scripts = [path for path in tracked if path.suffix.lower() == ".sh"]
    if artifacts or shell_scripts:
        print("Forbidden tracked files:", file=sys.stderr)
        for path in artifacts + shell_scripts:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    entrypoints = public_entrypoints()
    for entrypoint in entrypoints:
        relative = entrypoint.relative_to(ROOT)
        process = subprocess.run(
            [sys.executable, str(relative), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if process.returncode:
            print(f"Entrypoint failed: {relative}", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
            return process.returncode
        print(f"ok  {relative}")

    print(f"validated {len(entrypoints)} paper-facing entrypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
