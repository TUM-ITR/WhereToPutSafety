"""
generate disturbance sequences for the simulation
Ensures fair comparison: all controllers use identical noise per trial 
"""
import numpy as np
import pandas as pd
import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_module.params import SystemParams

def generate_single_disturbances(params: SystemParams, output_dir: str, seed: int = 0, verbose: bool = False,save_csv:bool = False):
    """
    Generate and save a single disturbance sequence, save it and return it
    
    """
    dt = params.TA
    n_steps = int(params.total_time / dt)
    
    # Noise configuration
    sigma = params.torque_disturbance_std
    bound = getattr(params, 'torque_disturbance_bound', 0.01)
    base_seed = params.random_seed
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    noise_files = []
    
    # Generate noise 
    rng = np.random.RandomState(seed)

    # Generate noise: shape (n_steps, 3) for 3 joints
    # Each step: generate Gaussian, then clip L2 norm to sqrt(3)*clip_bound
    w_raw = rng.normal(0, sigma, (n_steps, 3))
    #max_norm = np.sqrt(3) * clip_bound
    max_norm = bound
    w = np.empty_like(w_raw)
    for i in range(n_steps):
        norm = np.linalg.norm(w_raw[i])
        if norm > max_norm and norm > 0.0:
            w[i] = w_raw[i] * (max_norm / norm)
        #elif norm < 0.5 * sigma:
        #    w[i] = w_raw[i] * (0.5 * sigma / norm)
        else:
            w[i] = w_raw[i]
        
        # Save as .npy (faster to load)
        npy_file = os.path.join(output_dir, f"disturbances.npy")
        np.save(npy_file, w)
        
        # Also save first trial as CSV for inspection
        if save_csv:
            csv_file = os.path.join(output_dir, f"disturbance.csv")
            df = pd.DataFrame(w, columns=['w1', 'w2', 'w3'])
            df['step'] = np.arange(n_steps)
            df['time'] = np.arange(n_steps) * dt
            df['magnitude'] = np.linalg.norm(w, axis=1)
            df = df[['step', 'time', 'w1', 'w2', 'w3', 'magnitude']]
            df.to_csv(csv_file, index=False)
                
    # Statistics
    w_norms = np.linalg.norm(w, axis=1)
    if(verbose):
        print(f"Successfully created disturbances.")
        print(f"    Seed: {seed}")
        print(f"    Mean ||w||: {np.mean(w_norms):.6f} rad/s")
        print(f"    Max  ||w||: {np.max(w_norms):.6f} rad/s")
        print(f"    Saved: {npy_file}")
    return w

if __name__ == "__main__":
   params = SystemParams()
   generate_single_disturbances(params, output_dir=params.out_dir, seed =params.random_seed, verbose= False,save_csv = False)