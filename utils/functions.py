import numpy as np
import pickle

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