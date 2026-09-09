"""Run the Section VII held-out readout-free geometry analysis."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.readout_latent_geometry.value_heldout_geometry")
