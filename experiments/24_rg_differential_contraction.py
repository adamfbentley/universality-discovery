"""
Experiment 24: RG Differential Contraction - The Killer Plot

This is the CRITICAL experiment that validates the entire framework.

Goal:
    Track BOTH distances vs block size:
    - d(BD, KPZ)(b)  — within-class (should collapse)
    - d(EW, KPZ)(b)  — between-class (should persist)
    
    If EW↔KPZ stays ~constant while BD→KPZ collapses, we have geometric
    proof that RG contracts "irrelevant" directions while preserving
    "relevant" ones.

Theory:
    Universality as Geometric Contraction Under RG:
    - Implementation differences (discrete vs continuum) occupy 
      variance-dominated "irrelevant" directions
    - RG coarse-graining contracts these irrelevant directions
    - Universality class differences occupy "relevant" directions
      that survive or change predictably under RG
    
Expected Outcome:
    Two curves with different behavior:
    - BD→KPZ: Rapid collapse (90%+ reduction)
    - EW↔KPZ: Stays separated (slow decay or constant)

Date: January 20, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.spatial.distance import euclidean
from scipy.stats import sem

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.physics_simulation import GrowthModelSimulator

def extract_gradient_moments(h_history, L):
    """
    Extract 6D gradient moment features from 1D interface.
    
    For 1D interfaces, we compute:
    - Spatial gradient statistics
    - Second derivative (curvature) statistics
    - Height variance
    
    Same philosophy as Exp 20-23 but adapted for 1D.
    """
    # Take final configuration
    h = h_history[-1]  # Shape: (L,)
    
    # Compute gradient (periodic boundaries) - MATCHING EXP 21
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
    
    # Compute second derivative (discrete Laplacian in 1D) - MATCHING EXP 21
    lap = np.roll(h, -1) + np.roll(h, 1) - 2*h
    
    # Gradient statistics (standardized moments) - MATCHING EXP 21
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)
    
    if grad_std > 1e-10:
        grad_skew = np.mean((grad - grad_mean)**3) / grad_std**3
        grad_kurt = np.mean((grad - grad_mean)**4) / grad_std**4 - 3
    else:
        grad_skew = 0
        grad_kurt = 0
    
    # Gradient-Laplacian covariance - MATCHING EXP 21
    grad_lap_cov = np.mean((grad - np.mean(grad)) * (lap - np.mean(lap)))
    
    # Height variance - MATCHING EXP 21
    h_var = np.var(h - np.mean(h))
    
    # 6D feature vector (EXACTLY matching Exp 21)
    features = np.array([
        np.var(grad),             # grad_var (NOT grad_sq!)
        grad_skew,                # grad_skew (standardized)
        grad_kurt,                # grad_kurt (standardized)
        np.var(lap),              # lap_var
        grad_lap_cov,             # grad_lap_cov
        h_var                     # h_var
    ])
    
    return features

def apply_block_rg(h_history, block_size):
    """
    Apply proper block RG transformation to 1D interface.
    
    Block averaging in space.
    Returns coarse-grained height history.
    """
    if block_size == 1:
        return h_history
    
    # Take only configurations divisible by block_size
    L = h_history.shape[1]
    L_new = L // block_size
    
    # Block average
    h_coarse = np.zeros((h_history.shape[0], L_new))
    
    for t in range(h_history.shape[0]):
        h = h_history[t]
        for i in range(L_new):
            block = h[i*block_size:(i+1)*block_size]
            h_coarse[t, i] = np.mean(block)
    
    return h_coarse

def compute_pairwise_distance(features1, features2, whiten_params=None):
    """
    Compute Euclidean distance between feature sets.
    
    If whiten_params provided, apply whitening first.
    Returns mean distance across samples.
    """
    if whiten_params is not None:
        mean, std = whiten_params
        features1 = (features1 - mean) / (std + 1e-10)
        features2 = (features2 - mean) / (std + 1e-10)
    
    # Compute pairwise distances
    distances = []
    for f1 in features1:
        for f2 in features2:
            distances.append(euclidean(f1, f2))
    
    return np.mean(distances), np.std(distances)

def main():
    """Run Experiment 24: The Killer Plot."""
    
    print("=" * 70)
    print("Experiment 24: RG Differential Contraction")
    print("=" * 70)
    print()
    print("This is THE critical experiment.")
    print("Testing whether RG contracts implementation differences")
    print("while preserving universality class separation.")
    print()
    
    # Parameters - ENHANCED for larger scales
    L = 512  # System size (INCREASED from 256)
    T = 3000  # Simulation time (INCREASED from 2000)
    n_samples = 40  # Samples per model (DOUBLED from 20)
    block_sizes = [1, 2, 4, 8, 16, 32]  # RG scales to test (ADDED b=32)
    
    print(f"Parameters:")
    print(f"  L = {L}")
    print(f"  T = {T}")
    print(f"  n_samples = {n_samples}")
    print(f"  block_sizes = {block_sizes}")
    print()
    
    # Storage for results
    results = {
        'block_sizes': block_sizes,
        'bd_kpz_distances': [],
        'bd_kpz_errors': [],
        'ew_kpz_distances': [],
        'ew_kpz_errors': [],
        'bd_ew_distances': [],  # Bonus: track this too
        'bd_ew_errors': []
    }
    
    # Generate data for each model
    print("Phase 1: Generating simulation data...")
    print("-" * 70)
    
    models_data = {}
    
    # Initialize simulator
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    
    for model_name in ['EW', 'KPZ', 'BD']:
        print(f"Generating {model_name}...")
        histories = []
        
        for i in range(n_samples):
            if i % 10 == 0:  # Print every 10 samples
                print(f"  Sample {i+1}/{n_samples}")
            
            # Map model names to simulator types
            if model_name == 'EW':
                model_type = 'edwards_wilkinson'
                kwargs = {'diffusion': 1.0, 'noise_strength': 1.0, 'dt': 0.1}
            elif model_name == 'KPZ':
                model_type = 'kpz_equation'
                kwargs = {'diffusion': 1.0, 'nonlinearity': 1.0, 'noise_strength': 1.0, 'dt': 0.05}
            elif model_name == 'BD':
                model_type = 'ballistic_deposition'
                kwargs = {'noise_strength': 0.2}
            
            # Generate trajectory (returns shape (T, L))
            trajectory = simulator.generate_trajectory(model_type, **kwargs)
            
            # Reshape to 3D for consistency: (time_steps, L, L) where second L is 1
            # Actually, keep as 2D (T, L) and adjust feature extraction
            histories.append(trajectory)
        
        models_data[model_name] = histories
        print(f"  ✓ {model_name} complete ({len(histories)} samples)")
        print()
    
    # Process each block size
    print("\nPhase 2: Computing distances at each RG scale...")
    print("-" * 70)
    
    for b_idx, block_size in enumerate(block_sizes):
        print(f"\nBlock size b = {block_size}:")
        
        # Apply RG and extract features for all models
        all_features = {}
        
        for model_name in ['EW', 'KPZ', 'BD']:
            features_list = []
            
            for h_history in models_data[model_name]:
                # Apply RG
                h_coarse = apply_block_rg(h_history, block_size)
                
                # Extract features (use coarse-grained system size)
                L_coarse = h_coarse.shape[1]
                features = extract_gradient_moments(h_coarse, L_coarse)
                features_list.append(features)
            
            all_features[model_name] = np.array(features_list)
            print(f"  {model_name}: features shape = {all_features[model_name].shape}")
        
        # Compute whitening parameters from ALL data at this scale
        all_combined = np.vstack([
            all_features['EW'],
            all_features['KPZ'],
            all_features['BD']
        ])
        whiten_mean = np.mean(all_combined, axis=0)
        whiten_std = np.std(all_combined, axis=0)
        whiten_params = (whiten_mean, whiten_std)
        
        print(f"  Whitening: mean = {whiten_mean[:3]}... std = {whiten_std[:3]}...")
        
        # Compute key distances with whitening
        bd_kpz_dist, bd_kpz_err = compute_pairwise_distance(
            all_features['BD'], 
            all_features['KPZ'],
            whiten_params
        )
        
        ew_kpz_dist, ew_kpz_err = compute_pairwise_distance(
            all_features['EW'],
            all_features['KPZ'],
            whiten_params
        )
        
        bd_ew_dist, bd_ew_err = compute_pairwise_distance(
            all_features['BD'],
            all_features['EW'],
            whiten_params
        )
        
        # Store results
        results['bd_kpz_distances'].append(bd_kpz_dist)
        results['bd_kpz_errors'].append(bd_kpz_err / np.sqrt(n_samples))
        results['ew_kpz_distances'].append(ew_kpz_dist)
        results['ew_kpz_errors'].append(ew_kpz_err / np.sqrt(n_samples))
        results['bd_ew_distances'].append(bd_ew_dist)
        results['bd_ew_errors'].append(bd_ew_err / np.sqrt(n_samples))
        
        print(f"  d(BD, KPZ) = {bd_kpz_dist:.4f} ± {bd_kpz_err/np.sqrt(n_samples):.4f}")
        print(f"  d(EW, KPZ) = {ew_kpz_dist:.4f} ± {ew_kpz_err/np.sqrt(n_samples):.4f}")
        print(f"  d(BD, EW)  = {bd_ew_dist:.4f} ± {bd_ew_err/np.sqrt(n_samples):.4f}")
    
    print("\n" + "=" * 70)
    print("Phase 3: Analysis and Visualization")
    print("=" * 70)
    
    # Compute contraction ratios
    bd_kpz_initial = results['bd_kpz_distances'][0]
    bd_kpz_final = results['bd_kpz_distances'][-1]
    bd_kpz_reduction = (bd_kpz_initial - bd_kpz_final) / bd_kpz_initial * 100
    
    ew_kpz_initial = results['ew_kpz_distances'][0]
    ew_kpz_final = results['ew_kpz_distances'][-1]
    ew_kpz_change = (ew_kpz_initial - ew_kpz_final) / ew_kpz_initial * 100
    
    print(f"\nContraction Analysis:")
    print(f"  BD→KPZ: {bd_kpz_initial:.4f} → {bd_kpz_final:.4f} ({bd_kpz_reduction:+.1f}% change)")
    print(f"  EW↔KPZ: {ew_kpz_initial:.4f} → {ew_kpz_final:.4f} ({ew_kpz_change:+.1f}% change)")
    print()
    
    # Determine outcome
    if bd_kpz_reduction > 70 and abs(ew_kpz_change) < 30:
        print("✅ FRAMEWORK VALIDATED:")
        print("   - Within-class distance (BD→KPZ) contracts strongly")
        print("   - Between-class distance (EW↔KPZ) persists")
        print("   - RG selectively contracts irrelevant (implementation) directions")
        outcome = "SUCCESS"
    elif bd_kpz_reduction > 50:
        print("⚠️ PARTIAL VALIDATION:")
        print("   - BD→KPZ contracts, but not dramatically")
        print("   - May need larger block sizes or different observables")
        outcome = "PARTIAL"
    else:
        print("❌ FRAMEWORK NOT VALIDATED:")
        print("   - Both distances change similarly")
        print("   - No differential contraction observed")
        outcome = "FAILURE"
    
    print()
    
    # Create THE KILLER PLOT
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main plot: Distance vs Block Size
    ax1.errorbar(
        block_sizes,
        results['bd_kpz_distances'],
        yerr=results['bd_kpz_errors'],
        marker='o',
        markersize=8,
        linewidth=2,
        capsize=5,
        label='d(BD, KPZ) — Within-class',
        color='#E74C3C'
    )
    
    ax1.errorbar(
        block_sizes,
        results['ew_kpz_distances'],
        yerr=results['ew_kpz_errors'],
        marker='s',
        markersize=8,
        linewidth=2,
        capsize=5,
        label='d(EW, KPZ) — Between-class',
        color='#3498DB'
    )
    
    ax1.set_xlabel('Block Size b', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Euclidean Distance (whitened)', fontsize=12, fontweight='bold')
    ax1.set_title('RG Differential Contraction: The Killer Plot', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Annotate reduction percentages
    ax1.annotate(
        f'{bd_kpz_reduction:.0f}% reduction',
        xy=(block_sizes[-1], bd_kpz_final),
        xytext=(10, 20),
        textcoords='offset points',
        fontsize=10,
        color='#E74C3C',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#E74C3C')
    )
    
    ax1.annotate(
        f'{ew_kpz_change:+.0f}% change',
        xy=(block_sizes[-1], ew_kpz_final),
        xytext=(10, -30),
        textcoords='offset points',
        fontsize=10,
        color='#3498DB',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#3498DB')
    )
    
    # Second plot: Normalized distances (both start at 1.0)
    bd_kpz_normalized = np.array(results['bd_kpz_distances']) / bd_kpz_initial
    ew_kpz_normalized = np.array(results['ew_kpz_distances']) / ew_kpz_initial
    
    ax2.plot(
        block_sizes,
        bd_kpz_normalized,
        marker='o',
        markersize=8,
        linewidth=2,
        label='d(BD, KPZ) — Within-class',
        color='#E74C3C'
    )
    
    ax2.plot(
        block_sizes,
        ew_kpz_normalized,
        marker='s',
        markersize=8,
        linewidth=2,
        label='d(EW, KPZ) — Between-class',
        color='#3498DB'
    )
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial distance')
    ax2.set_xlabel('Block Size b', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Distance (d/d₀)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Contraction', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / 'results' / 'exp24_rg_differential_contraction'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / 'killer_plot.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    
    # Save numerical results
    results_path = output_dir / 'distances.txt'
    with open(results_path, 'w') as f:
        f.write("Experiment 24: RG Differential Contraction\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Parameters: L={L}, T={T}, n_samples={n_samples}\n\n")
        f.write("Distance Evolution:\n")
        f.write("-" * 70 + "\n")
        f.write("b\t\td(BD,KPZ)\td(EW,KPZ)\td(BD,EW)\n")
        f.write("-" * 70 + "\n")
        
        for i, b in enumerate(block_sizes):
            f.write(f"{b}\t\t{results['bd_kpz_distances'][i]:.4f}\t\t"
                   f"{results['ew_kpz_distances'][i]:.4f}\t\t"
                   f"{results['bd_ew_distances'][i]:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write(f"BD→KPZ contraction: {bd_kpz_reduction:.1f}%\n")
        f.write(f"EW↔KPZ change: {ew_kpz_change:+.1f}%\n")
        f.write(f"\nOutcome: {outcome}\n")
        
        if outcome == "SUCCESS":
            f.write("\n✅ FRAMEWORK VALIDATED\n")
            f.write("Universality emerges as RG contracts implementation differences\n")
            f.write("while preserving class separation.\n")
    
    print(f"✓ Saved: {results_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 24 COMPLETE")
    print("=" * 70)
    print(f"\nOutcome: {outcome}")
    print(f"\nKey Result:")
    print(f"  - BD→KPZ (within-class): {bd_kpz_reduction:.1f}% reduction")
    print(f"  - EW↔KPZ (between-class): {ew_kpz_change:+.1f}% change")
    
    if outcome == "SUCCESS":
        print("\n🎯 THIS IS THE RESULT WE NEEDED.")
        print("   The framework is validated: RG contracts irrelevant directions")
        print("   while preserving universality class structure.")
        print("\n   This single plot captures the entire theoretical framework.")
    
    print(f"\nOutputs saved to: {output_dir}")
    print()

if __name__ == "__main__":
    main()
