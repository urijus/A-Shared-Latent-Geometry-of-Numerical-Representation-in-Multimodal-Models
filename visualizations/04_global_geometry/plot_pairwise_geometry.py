"""Plot the Section VII pairwise exact-L numerical geometry."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualizations._entrypoint import run

if __name__ == "__main__":
    run("visualizations/main_paper/plot_causal_L_shared_numerical_geometry.py")
