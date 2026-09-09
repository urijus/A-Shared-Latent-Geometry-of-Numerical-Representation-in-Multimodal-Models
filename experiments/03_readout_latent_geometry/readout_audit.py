"""Run the Section VII digit-readout and exact C/L decomposition audit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.readout_latent_geometry.readout_causal_audit")
