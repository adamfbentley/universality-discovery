"""
Diagnostic analysis for Experiment 55 (Potts FSS failure).
Investigates why FSS collapse failed to recover ν = 5/6.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

print("="*70)
print("EXPERIMENT 55 DIAGNOSTICS: Why did FSS fail?")
print("="*70)

# Load data
try:
    data = np.load('data/exp55_potts_data.npz')
    features = data['features']
    sizes = data['sizes']
    temperatures = data['temperatures']
    T_c = data['T_c']
    print(f"\n✓ Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"  Sizes: {sizes}")
    print(f"  T_c = {T_c:.6f}")
except FileNotFoundError:
    print("\n❌ Data file not found: data/exp55_potts_data.npz")
    print("   Run experiments/55_potts_fss.py first")
    sys.exit(1)

# Reshape data
n_sizes = len(sizes)
n_temps = len(temperatures)
n_samples = features.shape[0] // (n_sizes * n_temps)
print(f"  {n_sizes} sizes × {n_temps} temps × {n_samples} samples = {features.shape[0]} total")

# Feature names
feature_names = ['Var(local_m)', '|∇m|', 'Var(|∇m|)', 'boundary', 'corr_1', 'entropy', 'cluster_var']

# ============================================================================
# DIAGNOSTIC 1: Raw observables vs temperature
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 1: RAW OBSERVABLE BEHAVIOR")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for feat_idx in range(7):
    ax = axes[feat_idx]
    
    for i, L in enumerate(sizes):
        # Get data for this size
        start_idx = i * n_temps * n_samples
        feat_vs_temp = []
        feat_std_vs_temp = []
        
        for j, T in enumerate(temperatures):
            idx_start = start_idx + j * n_samples
            idx_end = idx_start + n_samples
            vals = features[idx_start:idx_end, feat_idx]
            feat_vs_temp.append(np.mean(vals))
            feat_std_vs_temp.append(np.std(vals))
        
        feat_vs_temp = np.array(feat_vs_temp)
        feat_std_vs_temp = np.array(feat_std_vs_temp)
        
        # Plot
        ax.errorbar(temperatures, feat_vs_temp, yerr=feat_std_vs_temp, 
                    label=f'L={L}', marker='o', markersize=3, capsize=2, alpha=0.7)
    
    ax.axvline(T_c, color='red', linestyle='--', alpha=0.5, label='T_c')
    ax.set_xlabel('Temperature')
    ax.set_ylabel(feature_names[feat_idx])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[7].text(0.5, 0.5, 'Look for:\n• Peak/discontinuity at T_c\n• Size-dependent behavior\n• Crossing points', 
             ha='center', va='center', transform=axes[7].transAxes, fontsize=10)
axes[8].axis('off')

plt.tight_layout()
plt.savefig('figures/exp55_diagnostics_raw_observables.png', dpi=150, bbox_inches='tight')
print("  → Saved: figures/exp55_diagnostics_raw_observables.png")

# Check for critical behavior
print("\nCritical behavior check:")
for feat_idx, name in enumerate(feature_names):
    # Look for peak near T_c for largest system
    L_max_idx = n_sizes - 1
    start_idx = L_max_idx * n_temps * n_samples
    
    vals_vs_temp = []
    for j in range(n_temps):
        idx_start = start_idx + j * n_samples
        idx_end = idx_start + n_samples
        vals_vs_temp.append(np.mean(features[idx_start:idx_end, feat_idx]))
    
    vals_vs_temp = np.array(vals_vs_temp)
    
    # Find temperature of max value
    peak_idx = np.argmax(vals_vs_temp)
    peak_T = temperatures[peak_idx]
    
    # Check if peak is near T_c
    if abs(peak_T - T_c) < 0.02:
        print(f"  ✓ {name:15s}: Peak at T={peak_T:.4f} (near T_c)")
    else:
        print(f"  ✗ {name:15s}: Peak at T={peak_T:.4f} (far from T_c={T_c:.4f})")

# ============================================================================
# DIAGNOSTIC 2: PCA structure
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 2: PCA STRUCTURE")
print("="*70)

# Standardize features
features_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

# PCA
pca = PCA(n_components=7)
coords = pca.fit_transform(features_std)

print(f"\nExplained variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")

print(f"\nPC1 loadings (what does PC1 measure?):")
pc1_loadings = pca.components_[0]
for i, (name, loading) in enumerate(zip(feature_names, pc1_loadings)):
    print(f"  {name:15s}: {loading:+.4f}")

# Plot PC1 loadings
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_names, pc1_loadings)
ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('PC1 Loading')
ax.set_title('PC1 Composition (What is PC1 measuring?)')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figures/exp55_diagnostics_pc1_loadings.png', dpi=150, bbox_inches='tight')
print("\n  → Saved: figures/exp55_diagnostics_pc1_loadings.png")

# ============================================================================
# DIAGNOSTIC 3: PC1 vs temperature (does it show critical behavior?)
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 3: PC1 CRITICAL BEHAVIOR")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot PC1 vs T for each size
for i, L in enumerate(sizes):
    start_idx = i * n_temps * n_samples
    
    pc1_vs_temp = []
    pc1_std_vs_temp = []
    
    for j, T in enumerate(temperatures):
        idx_start = start_idx + j * n_samples
        idx_end = idx_start + n_samples
        vals = coords[idx_start:idx_end, 0]  # PC1
        pc1_vs_temp.append(np.mean(vals))
        pc1_std_vs_temp.append(np.std(vals))
    
    pc1_vs_temp = np.array(pc1_vs_temp)
    pc1_std_vs_temp = np.array(pc1_std_vs_temp)
    
    # Left: PC1 vs T
    ax1.errorbar(temperatures, pc1_vs_temp, yerr=pc1_std_vs_temp, 
                label=f'L={L}', marker='o', markersize=4, capsize=3, alpha=0.7)
    
    # Right: PC1 vs reduced temperature
    t_reduced = (temperatures - T_c) / T_c
    ax2.errorbar(t_reduced, pc1_vs_temp, yerr=pc1_std_vs_temp, 
                label=f'L={L}', marker='o', markersize=4, capsize=3, alpha=0.7)

ax1.axvline(T_c, color='red', linestyle='--', alpha=0.5, label='T_c')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('PC1')
ax1.set_title('PC1 vs Temperature')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Reduced temperature (T - T_c) / T_c')
ax2.set_ylabel('PC1')
ax2.set_title('PC1 vs Reduced Temperature')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/exp55_diagnostics_pc1_behavior.png', dpi=150, bbox_inches='tight')
print("  → Saved: figures/exp55_diagnostics_pc1_behavior.png")

# Check for size-dependent crossing
print("\nSize-dependent behavior:")
print("  If PC1 curves cross at T_c, that's good for FSS")
print("  If they don't cross or trend monotonically, FSS will fail")

# ============================================================================
# DIAGNOSTIC 4: Comparison to Ising (if available)
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 4: COMPARISON TO ISING")
print("="*70)

try:
    ising_data = np.load('data/exp52d_ising_fss.npz')
    ising_features = ising_data['features']
    ising_sizes = ising_data['sizes']
    ising_temps = ising_data['temperatures']
    ising_Tc = ising_data['T_c']
    
    print(f"✓ Loaded Ising data for comparison")
    
    # PCA on Ising
    ising_features_std = (ising_features - ising_features.mean(axis=0)) / (ising_features.std(axis=0) + 1e-10)
    ising_pca = PCA(n_components=min(7, ising_features.shape[1]))
    ising_coords = ising_pca.fit_transform(ising_features_std)
    
    print(f"\nIsing PC1 variance: {ising_pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Potts PC1 variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
    
    # Compare PC1 loadings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Potts
    ax1.barh(feature_names, pc1_loadings, color='blue', alpha=0.7)
    ax1.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('PC1 Loading')
    ax1.set_title('Potts PC1 Composition')
    ax1.grid(alpha=0.3, axis='x')
    
    # Ising (may have different features)
    ising_feature_names = ['Var(local_m)', '|∇m|', 'Var(|∇m|)', 'boundary', 'corr_1', 'entropy', 'cluster_var'][:ising_features.shape[1]]
    ising_pc1 = ising_pca.components_[0]
    ax2.barh(ising_feature_names, ising_pc1, color='red', alpha=0.7)
    ax2.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('PC1 Loading')
    ax2.set_title('Ising PC1 Composition')
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/exp55_diagnostics_potts_vs_ising.png', dpi=150, bbox_inches='tight')
    print("  → Saved: figures/exp55_diagnostics_potts_vs_ising.png")
    
except FileNotFoundError:
    print("✗ Ising data not found (exp52d_ising_fss.npz)")
    print("  Cannot compare to successful Ising experiment")

# ============================================================================
# DIAGNOSTIC 5: Manual FSS attempt with best observable
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 5: WHICH OBSERVABLE IS BEST?")
print("="*70)

print("\nTesting FSS collapse quality for each raw observable:")

def fss_collapse_quality(observable_data, sizes, temps, T_c, nu_test):
    """Test FSS collapse quality for a given observable and nu."""
    n_sizes = len(sizes)
    n_temps = len(temps)
    
    # Rescale for each size
    all_x = []
    all_y = []
    
    for i, L in enumerate(sizes):
        start_idx = i * n_temps * n_samples
        
        for j, T in enumerate(temps):
            idx_start = start_idx + j * n_samples
            idx_end = idx_start + n_samples
            
            obs_vals = observable_data[idx_start:idx_end]
            obs_mean = np.mean(obs_vals)
            
            # FSS variables
            t = (T - T_c) / T_c
            x_scaled = t * L**(1/nu_test)
            y_scaled = obs_mean / L  # Simple scaling (could adjust exponent)
            
            all_x.append(x_scaled)
            all_y.append(y_scaled)
    
    # Quality: inverse of variance in y for bins of x
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Bin and compute spread
    x_bins = np.linspace(all_x.min(), all_x.max(), 10)
    spreads = []
    for i in range(len(x_bins)-1):
        mask = (all_x >= x_bins[i]) & (all_x < x_bins[i+1])
        if mask.sum() > 3:
            spreads.append(np.std(all_y[mask]))
    
    if len(spreads) > 0:
        return np.mean(spreads)
    else:
        return np.inf

nu_exact = 5/6
nu_test_vals = [0.6, 0.7, 0.8, 0.833, 0.9, 1.0, 1.1]

for feat_idx, name in enumerate(feature_names):
    print(f"\n{name}:")
    qualities = []
    for nu_test in nu_test_vals:
        qual = fss_collapse_quality(features[:, feat_idx], sizes, temperatures, T_c, nu_test)
        qualities.append(qual)
    
    best_idx = np.argmin(qualities)
    best_nu = nu_test_vals[best_idx]
    print(f"  Best ν: {best_nu:.3f} (exact: {nu_exact:.3f}, error: {abs(best_nu - nu_exact)/nu_exact * 100:.1f}%)")
    
    if abs(best_nu - nu_exact) / nu_exact < 0.15:
        print(f"  ✓ Within 15% of exact value!")
    else:
        print(f"  ✗ Outside 15% tolerance")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print("""
Key questions answered by diagnostics:

1. Do raw observables show critical behavior at T_c?
   → Check exp55_diagnostics_raw_observables.png

2. What is PC1 actually measuring?
   → Check exp55_diagnostics_pc1_loadings.png

3. Does PC1 have the right temperature dependence for FSS?
   → Check exp55_diagnostics_pc1_behavior.png

4. How does this differ from successful Ising experiment?
   → Check exp55_diagnostics_potts_vs_ising.png (if available)

5. Would a different observable work better?
   → Check console output above for per-feature FSS tests

Next steps:
- If observables show clear peaks at T_c → problem is in PC1 composition
- If observables are flat → problem is in data generation (sampling/thermalization)
- If specific raw observable works → use that instead of PC1
""")
