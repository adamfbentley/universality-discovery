"""
Experiment 27: Initial Condition Control Test

THE KILLER EXPERIMENT for validating the universality axis.

Critical Question:
Does PC1 separation persist across different initial conditions?

If yes → PC1 is a fundamental universality coordinate
If no → PC1 is IC-dependent (but this matches KPZ fixed point theory!)

Initial Conditions:
1. Flat: h(x,0) = 0 (GOE Tracy-Widom expected)
2. Droplet: h(x,0) = -|x - L/2| (GUE Tracy-Widom expected)  
3. Stationary: h(x,0) ~ Brownian bridge (Baik-Rains expected)

The KPZ fixed point theory predicts DIFFERENT universal distributions 
for different IC classes. If PC1 rotates, that's scientifically meaningful!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import simulation functions
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from numba import jit

def generate_flat_ic(L):
    """Flat initial condition: h(x,0) = 0"""
    return np.zeros(L)

def generate_droplet_ic(L):
    """Droplet/curved initial condition: h(x,0) = -|x - L/2|"""
    x = np.arange(L)
    return -np.abs(x - L/2)

def generate_stationary_ic(L):
    """Stationary/Brownian initial condition"""
    # Brownian bridge: cumsum of Gaussian noise, rescaled to have zero mean
    brownian = np.cumsum(np.random.randn(L))
    # Make it periodic (Brownian bridge)
    brownian = brownian - np.linspace(0, brownian[-1], L)
    # Zero mean
    brownian = brownian - np.mean(brownian)
    return brownian

@jit(nopython=True)
def simulate_ew_step(interface, diffusion=1.0, noise_strength=1.0, dt=0.05):
    """Single EW time step: dh/dt = ν∇²h + η"""
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        dhdt = diffusion * laplacian + noise
        new_interface[x] = center + dt * dhdt
    
    return new_interface

@jit(nopython=True)
def simulate_kpz_step(interface, diffusion=1.0, nonlinearity=1.0, noise_strength=1.0, dt=0.05):
    """Single KPZ time step: dh/dt = ν∇²h + (λ/2)(∇h)² + η"""
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        gradient = (right - left) / 2.0
        nonlinear_term = nonlinearity * 0.5 * gradient**2
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        dhdt = diffusion * laplacian + nonlinear_term + noise
        new_interface[x] = center + dt * dhdt
    
    return new_interface

def simulate_growth(L, T, model='EW', h0=None, nu=1.0, lam=1.0, D=1.0, dt=0.05):
    """Simulate surface growth from initial condition h0"""
    if h0 is None:
        h0 = np.zeros(L)
    
    interface = h0.copy()
    n_steps = int(T / dt)
    
    for _ in range(n_steps):
        if model == 'EW':
            interface = simulate_ew_step(interface, diffusion=nu, noise_strength=D, dt=dt)
        else:  # KPZ
            interface = simulate_kpz_step(interface, diffusion=nu, nonlinearity=lam, 
                                         noise_strength=D, dt=dt)
        # Remove global tilt
        interface = interface - np.mean(interface)
    
    return interface

def extract_gradient_moments(h_final, L):
    """
    Extract 6D gradient moment features (matching Exp 21).
    
    Features:
    1. grad_var: Variance of spatial gradient
    2. grad_skew: Skewness of gradient
    3. grad_kurt: Kurtosis of gradient  
    4. lap_var: Variance of Laplacian
    5. grad_lap_cov: Covariance between |grad| and Laplacian
    6. h_var: Variance of height field
    """
    # Periodic boundary gradients
    grad = np.gradient(h_final, edge_order=2)
    grad_periodic = (np.roll(h_final, -1) - np.roll(h_final, 1)) / 2.0
    
    # Laplacian with periodic BC
    lap = np.roll(h_final, -1) + np.roll(h_final, 1) - 2*h_final
    
    # Moments
    grad_var = np.var(grad_periodic)
    grad_mean = np.mean(grad_periodic)
    grad_skew = np.mean((grad_periodic - grad_mean)**3) / (grad_var**1.5 + 1e-10)
    grad_kurt = np.mean((grad_periodic - grad_mean)**4) / (grad_var**2 + 1e-10)
    
    lap_var = np.var(lap)
    
    # Covariance between gradient magnitude and Laplacian
    grad_lap_cov = np.cov(np.abs(grad_periodic), lap)[0, 1]
    
    h_var = np.var(h_final)
    
    return np.array([grad_var, grad_skew, grad_kurt, lap_var, grad_lap_cov, h_var])

def main():
    print("=" * 80)
    print("EXPERIMENT 27: INITIAL CONDITION CONTROL TEST")
    print("=" * 80)
    print()
    print("Testing whether PC1 universality axis is IC-independent")
    print()
    
    # Parameters
    L = 128
    T = 1000  # Moderate time (mix of transient and approach to stationary)
    n_samples = 30  # Per model per IC
    
    # EW and KPZ parameters
    nu = 1.0
    D = 1.0
    lam = 1.0  # KPZ nonlinearity
    
    ic_types = ['flat', 'droplet', 'stationary']
    models = ['EW', 'KPZ']
    
    results = {ic: {model: [] for model in models} for ic in ic_types}
    
    print(f"Parameters: L={L}, T={T}, n_samples={n_samples} per model per IC")
    print(f"Initial conditions: {ic_types}")
    print()
    
    # Generate data for each IC type
    for ic_type in ic_types:
        print(f"\n{'='*60}")
        print(f"Generating {ic_type.upper()} initial condition data")
        print('='*60)
        
        for model in models:
            print(f"\n  {model}...", end='', flush=True)
            
            for i in range(n_samples):
                # Generate initial condition
                if ic_type == 'flat':
                    h0 = generate_flat_ic(L)
                elif ic_type == 'droplet':
                    h0 = generate_droplet_ic(L)
                else:  # stationary
                    h0 = generate_stationary_ic(L)
                
                # Simulate
                h_final = simulate_growth(L, T, model=model, h0=h0, nu=nu, lam=lam, D=D)
                
                # Extract features from final state
                features = extract_gradient_moments(h_final, L)
                
                results[ic_type][model].append(features)
                
                if (i+1) % 10 == 0:
                    print(f" {i+1}", end='', flush=True)
            
            print(f" ✓ ({n_samples} samples)")
    
    # Convert to arrays
    for ic_type in ic_types:
        for model in models:
            results[ic_type][model] = np.array(results[ic_type][model])
    
    print("\n" + "="*80)
    print("ANALYSIS: PCA and Separation Tests")
    print("="*80)
    
    # Analysis for each IC separately
    feature_names = ['grad_var', 'grad_skew', 'grad_kurt', 'lap_var', 'grad_lap_cov', 'h_var']
    
    ic_results = {}
    
    for ic_type in ic_types:
        print(f"\n{'='*60}")
        print(f"IC: {ic_type.upper()}")
        print('='*60)
        
        # Combine EW and KPZ
        X_ew = results[ic_type]['EW']
        X_kpz = results[ic_type]['KPZ']
        X_combined = np.vstack([X_ew, X_kpz])
        
        # Labels (0=EW, 1=KPZ)
        labels = np.array([0]*len(X_ew) + [1]*len(X_kpz))
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Correlation between PC1 and model label
        r_pc1, p_pc1 = pearsonr(X_pca[:, 0], labels)
        r_pc2, p_pc2 = pearsonr(X_pca[:, 1], labels)
        
        # Store results
        ic_results[ic_type] = {
            'X_pca': X_pca,
            'labels': labels,
            'r_pc1': r_pc1,
            'p_pc1': p_pc1,
            'r_pc2': r_pc2,
            'pca': pca,
            'loadings': pca.components_
        }
        
        print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
        print(f"\nCorrelation with model label:")
        print(f"  PC1: r = {r_pc1:.4f} (p = {p_pc1:.2e})")
        print(f"  PC2: r = {r_pc2:.4f} (p = {p_pc2:.2e})")
        
        print(f"\nPC1 loadings:")
        for i, name in enumerate(feature_names):
            print(f"  {name:15s}: {pca.components_[0, i]:+.3f}")
        
        print(f"\nPC2 loadings:")
        for i, name in enumerate(feature_names):
            print(f"  {name:15s}: {pca.components_[1, i]:+.3f}")
        
        # Cohen's d effect size
        pc1_ew = X_pca[labels == 0, 0]
        pc1_kpz = X_pca[labels == 1, 0]
        pooled_std = np.sqrt((np.var(pc1_ew) + np.var(pc1_kpz)) / 2)
        cohens_d = (np.mean(pc1_kpz) - np.mean(pc1_ew)) / pooled_std
        print(f"\nCohen's d (PC1): {cohens_d:.2f}")
    
    # CROSS-IC COMPARISON
    print("\n" + "="*80)
    print("CROSS-IC COMPARISON: Does PC1 rotate or persist?")
    print("="*80)
    
    # Compare PC1 loadings across ICs
    print("\nPC1 loading similarity (cosine similarity):")
    for i, ic1 in enumerate(ic_types):
        for ic2 in ic_types[i+1:]:
            v1 = ic_results[ic1]['loadings'][0, :]
            v2 = ic_results[ic2]['loadings'][0, :]
            # Cosine similarity (account for sign ambiguity)
            cos_sim = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            print(f"  {ic1:10s} vs {ic2:10s}: {cos_sim:.4f}")
    
    # Compare correlation strengths
    print("\nPC1 correlation with model label:")
    for ic_type in ic_types:
        r = ic_results[ic_type]['r_pc1']
        print(f"  {ic_type:10s}: r = {r:.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Check consistency
    all_strong = all(abs(ic_results[ic]['r_pc1']) > 0.8 for ic in ic_types)
    all_similar = True
    for i, ic1 in enumerate(ic_types):
        for ic2 in ic_types[i+1:]:
            v1 = ic_results[ic1]['loadings'][0, :]
            v2 = ic_results[ic2]['loadings'][0, :]
            cos_sim = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if cos_sim < 0.8:
                all_similar = False
    
    if all_strong and all_similar:
        print("\n✅ RESULT: PC1 is IC-INDEPENDENT")
        print("\nThe universality axis persists across all initial conditions.")
        print("This suggests PC1 encodes a fundamental property (likely λ/ν ratio)")
        print("that is independent of initial condition class.")
    elif all_strong and not all_similar:
        print("\n⚠️ RESULT: PC1 is IC-DEPENDENT (but still separates)")
        print("\nThe universality axis ROTATES with initial condition.")
        print("This matches KPZ fixed point theory: different IC classes → different")
        print("universal distributions (GOE/GUE/Baik-Rains).")
        print("\nThis is actually DEEPER - it reveals IC-class structure!")
    else:
        print("\n❌ RESULT: PC1 separation is WEAK or INCONSISTENT")
        print("\nThe universality axis may be a transient/growth-regime phenomenon.")
        print("At T={}, we may be in or near stationary regime where separation".format(T))
        print("collapses due to Gaussian stationary slope statistics.")
    
    # VISUALIZATION
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)
    
    output_dir = Path(__file__).parent.parent / 'results' / 'exp27_ic_control'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # PC1-PC2 scatter for each IC
    for idx, ic_type in enumerate(ic_types):
        ax = plt.subplot(2, 3, idx + 1)
        
        X_pca = ic_results[ic_type]['X_pca']
        labels = ic_results[ic_type]['labels']
        r = ic_results[ic_type]['r_pc1']
        
        # Plot
        ax.scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], 
                   c='blue', alpha=0.6, s=50, label='EW', edgecolors='k', linewidths=0.5)
        ax.scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], 
                   c='red', alpha=0.6, s=50, label='KPZ', edgecolors='k', linewidths=0.5)
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(f'{ic_type.upper()} IC: r(PC1,label) = {r:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
    
    # PC1 loadings comparison
    ax = plt.subplot(2, 3, 4)
    x = np.arange(len(feature_names))
    width = 0.25
    for idx, ic_type in enumerate(ic_types):
        loadings = ic_results[ic_type]['loadings'][0, :]
        ax.bar(x + idx*width, loadings, width, label=ic_type, alpha=0.8)
    
    ax.set_xlabel('Feature', fontsize=11)
    ax.set_ylabel('PC1 Loading', fontsize=11)
    ax.set_title('PC1 Loadings Across ICs', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', linewidth=0.5)
    
    # Correlation comparison
    ax = plt.subplot(2, 3, 5)
    correlations = [abs(ic_results[ic]['r_pc1']) for ic in ic_types]
    colors = ['green' if abs(ic_results[ic]['r_pc1']) > 0.8 else 'orange' for ic in ic_types]
    bars = ax.bar(ic_types, correlations, color=colors, alpha=0.7, edgecolor='k')
    ax.set_ylabel('|r(PC1, model)|', fontsize=11)
    ax.set_title('PC1 Separation Strength', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(0.8, color='r', linestyle='--', linewidth=1, label='Strong (>0.8)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Loading similarity heatmap
    ax = plt.subplot(2, 3, 6)
    n_ic = len(ic_types)
    sim_matrix = np.eye(n_ic)
    for i, ic1 in enumerate(ic_types):
        for j, ic2 in enumerate(ic_types):
            if i != j:
                v1 = ic_results[ic1]['loadings'][0, :]
                v2 = ic_results[ic2]['loadings'][0, :]
                sim_matrix[i, j] = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_ic))
    ax.set_yticks(range(n_ic))
    ax.set_xticklabels(ic_types)
    ax.set_yticklabels(ic_types)
    ax.set_title('PC1 Loading Cosine Similarity', fontsize=12, fontweight='bold')
    
    for i in range(n_ic):
        for j in range(n_ic):
            text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    fig_path = output_dir / 'ic_control_test.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {fig_path}")
    
    # Save numerical results
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT 27: INITIAL CONDITION CONTROL TEST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Parameters: L={L}, T={T}, n_samples={n_samples}\n\n")
        
        for ic_type in ic_types:
            f.write(f"\n{ic_type.upper()} IC:\n")
            f.write(f"  PC1 correlation: r = {ic_results[ic_type]['r_pc1']:.4f}\n")
            f.write(f"  PC1 loadings: {ic_results[ic_type]['loadings'][0, :]}\n")
        
        f.write("\n\nPC1 Loading Similarity:\n")
        for i, ic1 in enumerate(ic_types):
            for ic2 in ic_types[i+1:]:
                v1 = ic_results[ic1]['loadings'][0, :]
                v2 = ic_results[ic2]['loadings'][0, :]
                cos_sim = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                f.write(f"  {ic1} vs {ic2}: {cos_sim:.4f}\n")
    
    print(f"✓ Saved: {summary_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 27 COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
