"""Molecule and Hamiltonian helpers for the NH3 VQE project."""

from __future__ import annotations

import numpy as np
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, TaperedQubitMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.units import DistanceUnit


def create_nh3_driver(
    bond_length: float = 1.0325,
    bond_angle: float = 107.8,
    basis: str = "sto3g",
) -> PySCFDriver:
    """Create a PySCF driver for a symmetric NH3 geometry."""
    if not 0.5 <= bond_length <= 3.0:
        raise ValueError("bond_length must be in [0.5, 3.0] Angstrom")
    if not 90.0 <= bond_angle <= 120.0:
        raise ValueError("bond_angle must be in [90.0, 120.0] degrees")

    angle_rad = np.deg2rad(bond_angle)
    h_plane_z = -bond_length * np.cos(angle_rad / 2)
    r_xy = bond_length * np.sin(angle_rad / 2)

    hydrogens = []
    for idx in range(3):
        theta = 2 * np.pi * idx / 3
        hydrogens.append(
            (
                r_xy * np.cos(theta),
                r_xy * np.sin(theta),
                h_plane_z,
            )
        )

    atom_string = "N 0.0 0.0 0.0; " + "; ".join(
        f"H {x:.6f} {y:.6f} {z:.6f}" for x, y, z in hydrogens
    )
    return PySCFDriver(
        atom=atom_string,
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )


def get_qubit_operator(
    bond_length: float = 1.0325,
    bond_angle: float = 107.8,
    basis: str = "sto3g",
    freeze_core: bool = True,
):
    """Build the tapered qubit Hamiltonian for NH3."""
    driver = create_nh3_driver(bond_length=bond_length, bond_angle=bond_angle, basis=basis)
    problem = driver.run()
    if freeze_core:
        problem = FreezeCoreTransformer(freeze_core=True).transform(problem)

    mapper = TaperedQubitMapper(ParityMapper(num_particles=problem.num_particles))
    qubit_operator = mapper.map(problem.second_q_ops()[0])
    return qubit_operator, problem, mapper


def exact_energy(qubit_operator, problem) -> float:
    """Compute the exact reference energy for a mapped molecular Hamiltonian."""
    solution = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_operator)
    result = problem.interpret(solution)
    return float(result.total_energies[0])
