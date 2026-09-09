"""Plot the Section VII digit-readout and exact C/L decomposition."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualizations._entrypoint import run

if __name__ == "__main__":
    run("visualizations/main_paper/plot_readout_decomposition_main.py")
