"""Run the Section V Fourier analysis of textual arithmetic states."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.arithmetic_reference.fourier_probes.text.fourier_probing")
