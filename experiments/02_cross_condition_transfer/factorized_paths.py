"""Run the Section VI modality/operation path-composition experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._entrypoint import run

if __name__ == "__main__":
    run("src.experiments.cross_condition_transfer.procrustes.factorized_paths.factorized_paths")
