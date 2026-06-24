# Variational Quantum Eigensolver for Ammonia

**Python | Qiskit Nature | PennyLane | Quantum Chemistry**

Ground-state energy estimation for ammonia (`NH3`) using a Variational Quantum Eigensolver workflow with Qiskit Nature and PennyLane.

## Overview

This project explores a VQE pipeline for molecular-energy estimation on ammonia. The code builds a parametric `NH3` molecular geometry, maps the electronic-structure problem to a qubit Hamiltonian, and compares variational results against exact eigensolver references where feasible.

The repository was originally developed as a notebook-first project and has been refactored into a more professional structure: reusable chemistry and plotting utilities live under `src/`, while the exploratory notebook remains available under `notebooks/`.

## Method

Core steps:

- construct `NH3` geometry from bond length and bond angle
- generate the molecular Hamiltonian with Qiskit Nature and PySCF
- optionally freeze core orbitals
- apply parity mapping and tapered qubit reduction
- run exact reference energy estimation with NumPy eigensolver
- inspect VQE convergence and energy error

## Repository Structure

- `src/vqe_nh3/`: reusable Python utilities for molecule setup, Hamiltonian mapping, result loading, and plotting
- `scripts/`: command-line entry points for reproducible checks
- `notebooks/`: exploratory notebook retained for inspection
- `figures/`: static figures used in the project
- `utils/`: original helper module kept for backward compatibility

## Setup

```bash
pip install -r requirements.txt
```

The chemistry stack depends on PySCF/Qiskit/PennyLane compatibility, so using a fresh environment is recommended.

## Run

Single geometry smoke check:

```bash
python scripts/run_single_point.py --bond-length 1.0325 --bond-angle 107.8
```

The script builds the tapered qubit Hamiltonian and prints the exact reference energy for the selected geometry.

## Limitations

This is a research/course project rather than a packaged quantum-chemistry library. Public benchmark tables and full environment pinning are still limited; the next polish pass should add a small reproducible VQE run with saved outputs and a result table.
