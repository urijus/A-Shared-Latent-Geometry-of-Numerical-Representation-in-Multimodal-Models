"""Run the Section VII global frame synchronization experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.global_geometry.synchronization.synchronization_L")
