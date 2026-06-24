#!/usr/bin/env python3
"""Run a single NH3 exact-energy smoke check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqe_nh3 import exact_energy, get_qubit_operator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-length", type=float, default=1.0325)
    parser.add_argument("--bond-angle", type=float, default=107.8)
    parser.add_argument("--basis", default="sto3g")
    args = parser.parse_args()

    qubit_operator, problem, _ = get_qubit_operator(
        bond_length=args.bond_length,
        bond_angle=args.bond_angle,
        basis=args.basis,
    )
    energy = exact_energy(qubit_operator, problem)
    print(f"Qubits: {qubit_operator.num_qubits}")
    print(f"Pauli terms: {len(qubit_operator)}")
    print(f"Exact reference energy: {energy:.10f} Ha")


if __name__ == "__main__":
    main()
