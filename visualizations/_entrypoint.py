"""Run an established plotting script from a paper-oriented entry point."""

from __future__ import annotations

import runpy
from pathlib import Path


def run(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / relative_path), run_name="__main__")
