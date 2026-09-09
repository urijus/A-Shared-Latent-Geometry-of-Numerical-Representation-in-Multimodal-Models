"""Run the Section VIII latent-to-next-digit readout analysis."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.autoregressive_transfer.next_step_readout")
