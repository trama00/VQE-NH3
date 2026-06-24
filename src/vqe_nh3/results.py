"""Result-loading helpers for saved VQE computations."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_vqe_results(path: str | Path):
    """Load a pickled VQE result bundle produced by the notebook workflow."""
    result_path = Path(path)
    with result_path.open("rb") as handle:
        computation_results = pickle.load(handle)

    distances = np.array(computation_results["metadata"]["distances"])
    exact_energies = np.array(computation_results["exact_energies"])
    vqe_energies = np.array(computation_results["vqe_energies"])
    return computation_results, distances, exact_energies, vqe_energies
