"""Plot the Section VIII next-step autoregressive readout analysis."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualizations._entrypoint import run

if __name__ == "__main__":
    run("visualizations/main_paper/plot_latent_to_autoregressive_readout.py")
