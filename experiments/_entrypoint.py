"""Compatibility runner for the paper-oriented experiment entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run(module: str) -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    runpy.run_module(module, run_name="__main__")
