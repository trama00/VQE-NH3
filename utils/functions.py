import numpy as np
import pickle
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import ParityMapper, TaperedQubitMapper
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.quantum_info import Z2Symmetries
import matplotlib.pyplot as plt
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult
import time
from qiskit.quantum_info import SparsePauliOp
import json
import os
from typing import Dict, Any, Optional
import pennylane as qml


def load_vqe_results(filename, results_dir='results'):
    """
    Load previously saved VQE results
    
    Parameters:
    -----------
    filename : str, optional
        Specific filename to load (without path), if None loads the most recent
    results_dir : str, optional
        Directory to search for results files (default: 'results')
    
    Returns:
    --------
    tuple: (computation_results, distances, exact_energies, vqe_energies)
        computation_results : dict with complete results and metadata
        distances : numpy array of bond distances
        exact_energies : numpy array of exact energies
        vqe_energies : numpy array of VQE energies
    """
    # Handle both relative and absolute paths
    if not filename.startswith(results_dir):
        filename = f'{results_dir}/{filename}'
    print(f"Loading results from: {filename}")

    with open(filename, 'rb') as f:
        computation_results = pickle.load(f)
    
    # Extract main arrays for convenience
    distances = np.array(computation_results['metadata']['distances'])
    exact_energies = np.array(computation_results['exact_energies'])
    vqe_energies = np.array(computation_results['vqe_energies'])
    
    print(f"Loaded results for {len(distances)} bond length configurations")
    print(f"Computation date: {computation_results['metadata']['computation_date']}")
    print(f"Molecule: {computation_results['metadata']['molecule']}")
    print(f"Ansatz: {computation_results['metadata']['ansatz']}")
    print(f"Optimizer: {computation_results['metadata']['optimizer']}")

    return computation_results, distances, exact_energies, vqe_energies



def create_nh3_driver(bond_length=1.0325, bond_angle=107.8, basis='sto3g'):
    """
    Create a parametric NH3 molecule driver with correct C3v geometry.

    Parameters:
    -----------
    bond_length : float, optional
        N-H bond length in Angstrom (default: 1.0325, range: 0.5-3.0)
    bond_angle : float, optional  
        H-N-H bond angle in degrees (default: 107.8, range: 90-120)
    basis : str, optional
        basis set (default: 'sto3g')
        
    Returns:
    --------
    PySCFDriver: Configured molecular driver
    
    Raises:
    -------
    ValueError: If parameters are outside reasonable ranges
    """
    # Input validation
    if not (0.5 <= bond_length <= 3.0):
        raise ValueError(f"bond_length {bond_length:.3f} outside reasonable range [0.5, 3.0] Å")
    if not (90.0 <= bond_angle <= 120.0):
        raise ValueError(f"bond_angle {bond_angle:.1f} outside reasonable range [90.0, 120.0] degrees")
    if basis not in ['sto3g', 'sto-3g', '6-31g', '6-311g']:
        print(f"Warning: basis '{basis}' may not be supported. Common options: sto3g, 6-31g")
    
    angle_rad = np.deg2rad(bond_angle)
    
    # Height of N above the H triangle plane
    h_plane_z = -bond_length * np.cos(angle_rad / 2)
    r_xy = bond_length * np.sin(angle_rad / 2)

    # N at origin (0, 0, 0)
    # Place three H atoms symmetrically in the xy-plane
    h_positions = []
    for i in range(3):
        theta = 2 * np.pi * i / 3  # 120° separation
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        z = h_plane_z
        h_positions.append((x, y, z))
    
    # Build atom string
    atom_string = f"N 0.0 0.0 0.0; "
    atom_string += "; ".join([f"H {x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in h_positions])

    try:
        driver = PySCFDriver(
            atom=atom_string,
            basis=basis,
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to create PySCF driver: {str(e)}")
    

def get_qubit_op(bond_length=1.0325, bond_angle=106.8, basis='sto3g', freeze_core=True, verbose=True):
    """
    Get the qubit operator for NH3 molecule at given geometry
    
    Parameters:
    bond_length: N-H bond length in Angstrom (default: 1.0325)
    bond_angle: H-N-H bond angle in degrees (default: 106.8)
    basis: basis set string (default: 'sto3g')
    
    Returns:
    tuple: (tapered_qubit_op, num_particles, num_spatial_orbitals, problem, tapered_mapper)
    """
    # Simple cache to avoid recomputation across runs
    global _QUBIT_OP_CACHE
    try:
        _QUBIT_OP_CACHE
    except NameError:
        _QUBIT_OP_CACHE = {}
    key = (round(float(bond_length), 6), round(float(bond_angle), 6), str(basis), freeze_core)
    if key in _QUBIT_OP_CACHE:
        return _QUBIT_OP_CACHE[key]

    # Create driver with given geometry
    driver = create_nh3_driver(bond_length=bond_length, bond_angle=bond_angle, basis=basis)
    if freeze_core:
        problem = FreezeCoreTransformer(freeze_core=True).transform(driver.run())
    else:
        problem = driver.run()
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals
    
    if verbose:
        print("="*60)
        print("TAPERED QUBIT MAPPER PROCESS")
        print("="*60)
        print(f"1. Original Problem Size:")
        print(f"   - Spatial orbitals: {num_spatial_orbitals}")
        print(f"   - Spin orbitals (N): {2 * num_spatial_orbitals}")
        print(f"   - Electrons: {num_particles}")
        print()
    
    # Create the base mapper
    base_mapper = ParityMapper(num_particles=problem.num_particles)
    
    # Create the TaperedQubitMapper - this automatically handles symmetry detection and tapering!
    tapered_mapper = TaperedQubitMapper(base_mapper)
    
    # Map the operator - this automatically applies tapering!
    tapered_qubit_op = tapered_mapper.map(problem.second_q_ops()[0])
    
    if verbose:
        print(f"2. After TaperedQubitMapper:")
        print(f"   - Qubit operator type: {type(tapered_qubit_op).__name__}")
        print(f"   - Number of qubits required: {tapered_qubit_op.num_qubits}")
        print(f"   - Number of terms in Hamiltonian: {len(tapered_qubit_op)}")
        
        # Show information about the symmetries that were found and used
        if hasattr(tapered_mapper, 'z2symmetries') and tapered_mapper.z2symmetries:
            z2sym = tapered_mapper.z2symmetries
            print(f"   - Symmetries found: {len(z2sym.symmetries)}")
            print(f"   - Qubits saved: {2 * num_spatial_orbitals - tapered_qubit_op.num_qubits}")
        print()
        
        print(f"3. Summary:")
        print(f"   Jordan-Wigner would require: {2 * num_spatial_orbitals} qubits")
        print(f"   Tapered Parity requires: {tapered_qubit_op.num_qubits} qubits")
        print("="*60)

    # Return the tapered operator and mapper
    result = (tapered_qubit_op, num_particles, num_spatial_orbitals, problem, tapered_mapper)
    _QUBIT_OP_CACHE[key] = result
    return result

    
def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result


def plot_convergence(counts, values, ref_value, label):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(counts, values, label=label, alpha=0.7)
    plt.axhline(ref_value, color='red', linestyle='--', label=f'Reference: {ref_value:.5f}')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Energy (Ha)')
    plt.title('VQE Convergence - Energy vs Evaluations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    errors = np.abs(np.array(values) - ref_value)
    plt.semilogy(counts, errors, label=label, alpha=0.7)
    plt.xlabel('Function Evaluations')
    plt.ylabel('|Error| (Ha)')
    plt.title('VQE Convergence - Error vs Evaluations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
def interpret_values(raw_values, problem):
    """Convert raw eigenvalues to interpreted energies"""
    interpreted_values = []
    for raw_val in raw_values:
        # Create a simple result object
        temp_result = MinimumEigensolverResult()
        temp_result.eigenvalue = complex(raw_val)
        temp_result.eigenvalues = [complex(raw_val)]
        
        # Interpret using the quantum chemistry problem
        interpreted_result = problem.interpret(temp_result)
        interpreted_energy = interpreted_result.total_energies[0].real
        interpreted_values.append(interpreted_energy)
    
    return interpreted_values



def find_truncate_terms(qubit_op, problem, error_threshold, weight_threshold, verbose=True, plot=True):
    """
    Find number of terms to truncate Hamiltonian within error threshold
    Parameters:
        qubit_op (SparsePauliOp): Qubit Hamiltonian.
        problem (ElectronicStructureProblem): Electronic structure problem.
        error_threshold (float): Error threshold for truncation.
        weight_threshold (float): Weight threshold for truncation.
        verbose (bool): Whether to print progress.
        plot (bool): Whether to plot the results.
    Returns:
        k (int): Minimum number of terms to keep.
    """
    
    if error_threshold is None and weight_threshold is None:
        raise ValueError("At least one of error_threshold or weight_threshold must be provided")
    
    # --- Step 1: full energy ---
    solver = NumPyMinimumEigensolver()
    start_time = time.time()
    raw_energy = solver.compute_minimum_eigenvalue(qubit_op)
    energy = problem.interpret(raw_energy)
    energy = energy.total_energies[0].real
    full_time = time.time() - start_time
    if verbose:
        print(f"Full Hamiltonian energy: {energy:.8f} Ha, computed in {full_time:.1f} s")

    # --- Step 2: sort terms by |coeff| ---
    coeffs = qubit_op.coeffs
    paulis = qubit_op.paulis
    idx_sorted = np.argsort(-np.abs(coeffs))  # descending order

    # --- Step 3: iterate ---
    energies = []
    errors = []
    n_terms_list = []
    
    total_weight = np.sum(np.abs(coeffs))

    for k in range(1, len(coeffs) + 1):
        # Keep top-k terms
        selected = idx_sorted[:k]
        trunc_op = SparsePauliOp(paulis[selected], coeffs[selected])

        # Solve truncated Hamiltonian
        start_time = time.time()
        raw_energy_trunc = solver.compute_minimum_eigenvalue(trunc_op)
        energy_trunc = problem.interpret(raw_energy_trunc).total_energies[0].real
        trunc_time = time.time() - start_time
        
        # Store
        energies.append(energy_trunc)
        err = abs(energy_trunc - energy)
        errors.append(err)
        n_terms_list.append(k)
        
        # Weight fraction
        kept_weight = np.sum(np.abs(coeffs[selected]))
        weight_fraction = kept_weight / total_weight
        
        if verbose:
            print(f"k={k:3d}, Energy: {energy_trunc:.8f} Ha, Err: {abs(energy_trunc - energy)*1000:.3f} mHa, Weight kept: {weight_fraction:.3%}, Time: {trunc_time:.1f} s")

        # Optional stop at error_threshold
        if (error_threshold is None or err < error_threshold) and (weight_threshold is None or weight_fraction > weight_threshold):
            if verbose:
                print(f"Stopping at k={k} with error {err*1000:.3f} mHa and weight fraction {weight_fraction:.3%}")
            break

    if plot:
        # --- Step 4: plot ---
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(n_terms_list, energies, label="Truncated energy")
        plt.axhline(energy, color="r", linestyle="--", label="Full energy")
        plt.axvline(k, color="green", linestyle="-", label=f"k={k}")
        plt.xlabel("#Terms kept")
        plt.ylabel("Energy (Ha)")
        plt.legend()

        plt.subplot(1,2,2)
        plt.semilogy(n_terms_list, np.array(errors)*1000, label="Error")
        plt.axhline(1.6, color="r", linestyle="--", label="1.6 mHa")
        plt.axvline(k, color="green", linestyle="-", label=f"k={k}")
        plt.xlabel("#Terms kept")
        plt.ylabel("Error (mHa)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return k


def truncate_hamiltonian(qubit_op, num_terms, verbose=True):
    """
    Truncate Hamiltonian with detailed diagnostics.
    """
    coeffs = qubit_op.coeffs
    paulis = qubit_op.paulis
    
    # Sort by importance (absolute coefficient)
    idx_sorted = np.argsort(-np.abs(coeffs))
    selected_indices = idx_sorted[:num_terms]
    
    # Create truncated operator
    truncated_op = SparsePauliOp(paulis[selected_indices], coeffs[selected_indices])
    
    if verbose:
        total_weight = np.sum(np.abs(coeffs))
        kept_weight = np.sum(np.abs(coeffs[selected_indices]))
        weight_fraction = kept_weight / total_weight
        
        print(f"Truncation summary:")
        print(f"  Original terms: {len(coeffs)}")
        print(f"  Truncated terms: {num_terms}")
        print(f"  Weight kept: {weight_fraction:.3%}")
        print(f"  Largest term: {np.max(np.abs(coeffs[selected_indices])):.6f}")
        print(f"  Smallest kept term: {np.min(np.abs(coeffs[selected_indices])):.6f}")
    
    return truncated_op


def create_dict(ansatz_name, vqe_result, ref_value, counts, raw_values, parameters, dist, ang, k):
    return {
        "ansatz": ansatz_name,
        "vqe_result": vqe_result,
        "ref_value": ref_value,
        "counts": counts,
        "raw_values": raw_values,
        "parameters": parameters,
        "dist": dist,
        "ang": ang,
        "k": k
    }

def save_vqe_results(results, filename, new_params=False):
    """
    Save VQE results to a JSON file.
    
    Args:
        filename: Path to the JSON file
        results: Dictionary containing VQE results (counts, values, parameters, etc.)
        new_params: If True, change filename adding a progressive number to avoid overwriting
    -----------
    """
    try:
        if new_params:
            base, ext = os.path.splitext(filename)
            i = 2
            while os.path.exists(filename):
                filename = f"{base}_{i}{ext}"
                i += 1

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")
        
def load_vqe_results(filename, verbose=True):
    """
    Load VQE results from a JSON file.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        Dictionary containing VQE results or None if file doesn't exist/error
    """
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        if verbose:
            print(f"Results loaded from {filename}")
        return results
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None
    
def check_results_file(filename, dist, ang, k):
    """
    Check if a results file exists.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(filename) and \
           load_vqe_results(filename, verbose=False)['dist'] == dist and \
           load_vqe_results(filename, verbose=False)['ang'] == ang and \
           load_vqe_results(filename, verbose=False)['k'] == k
           
           
def to_qiskit_hamiltonian(hamiltonian, noq=None):
    """
    Convert a PennyLane Hamiltonian into a Qiskit SparsePauliOp.
    
    Args:
        hamiltonian (qml.Hamiltonian): PennyLane Hamiltonian
        noq (int): number of qubits
    
    Returns:
        SparsePauliOp: equivalent Qiskit Hamiltonian
    """
    
    OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}
    paulis = []
    coeffs = []
        
    for coeff, obs in zip(hamiltonian.terms()[0], hamiltonian.terms()[1]):
        pauli_term = ["I"] * noq
        
        # Handle different types of observables
        if isinstance(obs, qml.ops.op_math.Prod):
            # Product of observables
            for factor in obs.operands:
                if hasattr(factor, 'wires') and len(factor.wires) > 0:
                    pauli_term[factor.wires[0]] = OBS_MAP.get(factor.name, "I")
        elif hasattr(obs, 'name') and hasattr(obs, 'wires'):
            # Single observable
            if len(obs.wires) > 0:
                pauli_term[obs.wires[0]] = OBS_MAP.get(obs.name, "I")
        elif hasattr(obs, 'obs') and hasattr(obs, 'wires'):
            # Tensor product or other composite observables
            if len(obs.wires) > 0:
                # Handle tensor products
                if hasattr(obs, 'obs'):
                    for i, sub_obs in enumerate(obs.obs):
                        if i < len(obs.wires):
                            pauli_term[obs.wires[i]] = OBS_MAP.get(sub_obs.name, "I")
                else:
                    pauli_term[obs.wires[0]] = OBS_MAP.get(obs.name, "I")

        # Reverse because Qiskit uses little-endian convention
        term = "".join(pauli_term[::-1])
        paulis.append(term)
        coeffs.append(coeff)
    
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def generate_single_initializations(n_params, seed=42):
    """
    Generate a single parameter vector for each initialization scheme.

    Returns a dict with keys:
        'zeros'       : all zeros
        'small_normal': N(0,0.1)
        'uniform'     : Uniform(-π, π)
        'pm_pi_over2' : each parameter randomly +π/2 or -π/2
        'sobol'       : Sobol low-discrepancy point mapped to [-π, π]
    """
    rng = np.random.default_rng(seed)
    inits = {}

    inits['zeros'] = np.zeros(n_params)

    inits['small_normal'] = rng.normal(0.0, 0.1, n_params)

    inits['uniform'] = rng.uniform(-np.pi, np.pi, n_params)

    inits['pm_pi_over2'] = rng.choice([-np.pi/2 + rng.normal(0.0, 0.1), np.pi/2 + rng.normal(0.0, 0.1)], size=n_params)

    try:
        from scipy.stats import qmc
        sob = qmc.Sobol(d=n_params, scramble=True, seed=seed)
        u = sob.random(1)[0]        # one Sobol sample in [0,1]^n
        inits['sobol'] = np.pi * (2*u - 1)  # map to [-π, π]
    except Exception:
        # fall back to uniform if scipy isn't available
        inits['sobol'] = rng.uniform(-np.pi, np.pi, n_params)

    return inits


def draw_pl_ansatz(ansatz_fn, n_qubits, n_layers, dev, method="text"):
    """
    Draw a PennyLane ansatz circuit.

    Parameters:
    -----------
    ansatz_fn : function
        The ansatz function, taking (params, wires) as input.
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of layers in the ansatz.
    dev : pennylane.Device
        PennyLane device on which the circuit is defined.

    Returns:
    --------
    None (prints text-based circuit diagram)
    """
    # Wrap ansatz in a QNode
    @qml.qnode(dev)
    def circuit(params):
        ansatz_fn(params, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # Generate sample parameters
    params = np.random.rand(n_layers * n_qubits)

    if method == "text":
        drawer = qml.draw(circuit)
        print(drawer(params))
    elif method == "mpl":
        fig = qml.draw_mpl(circuit)(params)
        plt.show()
    else:
        raise ValueError("method must be 'text' or 'mpl'")
    
    
class pl_VQECallback:
    """Callback class to monitor VQE optimization progress"""
    
    def __init__(self, pl_ansatz_type, save_to_file=True):
        self.iteration = 0
        self.energies = []
        self.parameters = []
        self.times = []
        self.ansatz_type = pl_ansatz_type
        self.save_to_file = save_to_file
        self.start_time = time.time()
        
    def __call__(self, pl_params):
        """Callback function called at each iteration"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate current energy
        if hasattr(self, 'objective_func'):
            current_energy = self.objective_func(pl_params)
        else:
            # If objective function not available, use the last known energy
            current_energy = self.energies[-1] if self.energies else None
        
        # Store data
        self.energies.append(current_energy)
        self.parameters.append(pl_params.copy())
        self.times.append(elapsed_time)
        
        # Print progress
        if current_energy is not None:
            print(f"Iteration {self.iteration:3d}: Energy = {current_energy:.8f} Hartree, Time = {elapsed_time:.2f}s")
        else:
            print(f"Iteration {self.iteration:3d}: Time = {elapsed_time:.2f}s")
        
        # Save to file periodically (every 10 iterations)
        if self.save_to_file and self.iteration % 100 == 0:
            self.save_progress()
        
        self.iteration += 1
        
    def save_progress(self):
        """Save current progress to file"""
        filename = f"vqe_progress_{self.ansatz_type}_{int(time.time())}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        data = {
            'ansatz_type': self.ansatz_type,
            'iteration': self.iteration,
            'energies': [float(e) if e is not None else None for e in self.energies],
            'parameters': [p.tolist() if hasattr(p, 'tolist') else list(p) for p in self.parameters],
            'times': self.times,
            'total_iterations': len(self.energies)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Progress saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save progress to file: {e}")


def pl_plot_optimization_progress(pl_result_custom, pl_exact_energy=None):
    """Plot the optimization convergence for custom ansatz"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot energy convergence
    if hasattr(pl_result_custom, 'callback_data') and pl_result_custom.callback_data:
        pl_energies_custom = pl_result_custom.callback_data['energies']
        pl_iterations_custom = range(len(pl_energies_custom))
        ax1.plot(pl_iterations_custom, pl_energies_custom, 'b-', label='Custom Ansatz', linewidth=2)
        
        if pl_exact_energy is not None:
            ax1.axhline(y=pl_exact_energy, color='g', linestyle='--', linewidth=2, label='Exact Energy')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy (Hartree)')
        ax1.set_title('VQE Energy Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error vs exact energy (if available)
        if pl_exact_energy is not None:
            pl_errors_custom = [abs(e - pl_exact_energy) for e in pl_energies_custom if e is not None]
            ax2.semilogy(range(len(pl_errors_custom)), pl_errors_custom, 'b-', label='Custom Ansatz', linewidth=2)
            
            # Chemical accuracy line
            ax2.axhline(y=0.0016, color='orange', linestyle=':', linewidth=2, label='Chemical Accuracy')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Error (Hartree)')
            ax2.set_title('Error vs Exact Energy (Log Scale)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Exact energy not available\nfor error plotting', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Error Analysis Not Available')
    else:
        ax1.text(0.5, 0.5, 'No optimization data available\nRun optimization first', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Optimization Progress Not Available')
        
        ax2.text(0.5, 0.5, 'No optimization data available\nRun optimization first', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Error Analysis Not Available')
    
    plt.tight_layout()
    plt.show()
    
    
def save_q_results(qiskit_energy, qiskit_result, qiskit_counts, qiskit_values, qiskit_parameters, qiskit_optimization_time):
    """Save Qiskit VQE results to JSON file matching PennyLane format"""
    filename = f"vqe_progress_qiskit_{int(time.time())}.json"
    
    # Convert data to JSON-serializable format
    data = {
        'ansatz_type': 'qiskit_custom',
        'framework': 'qiskit',
        'iteration': len(qiskit_counts),
        'energies': [float(e) if e is not None else None for e in qiskit_values],
        'parameters': [p if isinstance(p, list) else list(p) for p in qiskit_parameters],
        'counts': qiskit_counts,  # Qiskit-specific: evaluation counts
        'total_iterations': len(qiskit_counts),
        'final_energy': float(qiskit_energy),
        'optimization_time': qiskit_optimization_time,
        'function_evaluations': qiskit_result.cost_function_evals,
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Qiskit VQE results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Warning: Could not save Qiskit results: {e}")
        return None