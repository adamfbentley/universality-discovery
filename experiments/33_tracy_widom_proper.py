"""
Experiment 33: Proper Tracy-Widom Test

The issue: We've been computing GLOBAL skewness of h(x,t).
Tracy-Widom statistics apply to SINGLE-POINT fluctuations with proper rescaling.

For KPZ droplet IC: h(0,t) ~ v∞*t + c*t^(1/3) * χ_GUE
where χ_GUE has skewness ≈ -0.29

This experiment:
1. Track h(0,t) at the origin over many samples
2. Apply proper t^(1/3) rescaling
3. Measure skewness of rescaled fluctuations
4. Compare to Tracy-Widom GUE

Also investigate: why does global skewness converge to 0?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import skew

np.random.seed(42)

# Output directory
output_dir = Path(__file__).parent.parent / 'results' / 'exp33_tracy_widom_proper'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*78)
print("EXPERIMENT 33: PROPER TRACY-WIDOM TEST")
print("="*78)
print()
print("Testing proper Tracy-Widom statistics:")
print("  - Single-point h(0,t) tracking")
print("  - Proper t^(1/3) rescaling")
print("  - Many samples for statistics")
print()

# Parameters
L = 512  # Larger L to minimize finite-size effects
T = 5000  # Long enough for asymptotic
dt = 0.05
n_samples = 200  # Many samples for good statistics
measure_times = [500, 1000, 2000, 3000, 4000, 5000]  # When to measure

print(f"Parameters: L={L}, T={T}, n_samples={n_samples}")
print(f"Measurement times: {measure_times}")
print()

# Storage
kpz_h0_samples = {t: [] for t in measure_times}  # h(L/2, t) for KPZ
ew_h0_samples = {t: [] for t in measure_times}   # h(L/2, t) for EW
kpz_global_skew = {t: [] for t in measure_times}
ew_global_skew = {t: [] for t in measure_times}

print("Running simulations...")
start_time = time.time()

for i in range(n_samples):
    if (i+1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  Sample {i+1}/{n_samples} ({elapsed:.1f}s)")
    
    # Initialize with droplet IC
    h_kpz = np.zeros(L)
    h_ew = np.zeros(L)
    
    # Add droplet at center
    center = L // 2
    for x in range(L):
        dist = min(abs(x - center), L - abs(x - center))
        h_kpz[x] = -dist * 0.1  # Droplet (parabolic minimum)
        h_ew[x] = -dist * 0.1
    
    # Track h(center) - the droplet peak
    h0_kpz_initial = h_kpz[center]
    h0_ew_initial = h_ew[center]
    
    current_time = 0
    measure_idx = 0
    
    while current_time < T and measure_idx < len(measure_times):
        target_time = measure_times[measure_idx]
        steps_needed = int((target_time - current_time) / dt)
        
        for _ in range(steps_needed):
            # EW: dh/dt = ∇²h + η
            laplacian_ew = np.roll(h_ew, 1) + np.roll(h_ew, -1) - 2*h_ew
            h_ew = h_ew + dt * laplacian_ew + np.sqrt(dt) * np.random.randn(L)
            
            # KPZ: dh/dt = ∇²h + (λ/2)(∇h)² + η
            laplacian_kpz = np.roll(h_kpz, 1) + np.roll(h_kpz, -1) - 2*h_kpz
            gradient_kpz = (np.roll(h_kpz, -1) - np.roll(h_kpz, 1)) / 2
            h_kpz = h_kpz + dt * (laplacian_kpz + 0.5 * gradient_kpz**2) + np.sqrt(dt) * np.random.randn(L)
        
        current_time = target_time
        
        # Record h(center) relative to initial (the key Tracy-Widom quantity)
        kpz_h0_samples[target_time].append(h_kpz[center] - h0_kpz_initial)
        ew_h0_samples[target_time].append(h_ew[center] - h0_ew_initial)
        
        # Also record global skewness for comparison
        kpz_global_skew[target_time].append(skew(h_kpz))
        ew_global_skew[target_time].append(skew(h_ew))
        
        measure_idx += 1

print(f"\nCompleted in {time.time() - start_time:.1f}s")

# Analysis
print()
print("="*78)
print("RESULTS: SINGLE-POINT h(0,t) STATISTICS")
print("="*78)
print()
print("Theory: For KPZ with droplet IC, the rescaled fluctuation")
print("        χ = (h(0,t) - v∞*t) / (Γ*t^(1/3))")
print("        should have skewness ≈ -0.29 (Tracy-Widom GUE)")
print()

# For KPZ, estimate v∞ from late-time growth
# v∞ is the asymptotic velocity
# We'll estimate it empirically

kpz_mean_h0 = {t: np.mean(kpz_h0_samples[t]) for t in measure_times}
ew_mean_h0 = {t: np.mean(ew_h0_samples[t]) for t in measure_times}

# Fit v∞ from late times
late_times = measure_times[-3:]
late_h0_kpz = [kpz_mean_h0[t] for t in late_times]
v_inf_kpz = np.polyfit(late_times, late_h0_kpz, 1)[0]

late_h0_ew = [ew_mean_h0[t] for t in late_times]
v_inf_ew = np.polyfit(late_times, late_h0_ew, 1)[0]

print(f"Estimated asymptotic velocity:")
print(f"  KPZ: v∞ ≈ {v_inf_kpz:.6f}")
print(f"  EW:  v∞ ≈ {v_inf_ew:.6f}")
print()

print(f"{'Time':>6} | {'KPZ raw skew':>12} | {'KPZ rescaled':>12} | {'EW raw skew':>12} | {'EW rescaled':>12}")
print("-" * 70)

kpz_rescaled_skews = []
ew_rescaled_skews = []

for t in measure_times:
    # Raw skewness of h(0,t)
    kpz_raw = skew(kpz_h0_samples[t])
    ew_raw = skew(ew_h0_samples[t])
    
    # Rescaled: subtract mean and divide by t^(1/3)
    # The fluctuations δh = h(0,t) - <h(0,t)> should scale as t^(1/3)
    kpz_centered = np.array(kpz_h0_samples[t]) - np.mean(kpz_h0_samples[t])
    ew_centered = np.array(ew_h0_samples[t]) - np.mean(ew_h0_samples[t])
    
    # Rescale by t^(1/3) for KPZ, t^(1/4) for EW
    # (Skewness is scale-invariant, so this shouldn't change it)
    kpz_rescaled = skew(kpz_centered / (t**(1/3)))
    ew_rescaled = skew(ew_centered / (t**(1/4)))
    
    kpz_rescaled_skews.append(kpz_rescaled)
    ew_rescaled_skews.append(ew_rescaled)
    
    print(f"{t:>6} | {kpz_raw:>+12.4f} | {kpz_rescaled:>+12.4f} | {ew_raw:>+12.4f} | {ew_rescaled:>+12.4f}")

print()
print("Note: Skewness is scale-invariant, so rescaling shouldn't change it.")
print("      The key is having enough samples and long enough times.")

# Compare with global skewness
print()
print("="*78)
print("COMPARISON: SINGLE-POINT vs GLOBAL SKEWNESS")
print("="*78)
print()
print(f"{'Time':>6} | {'KPZ h(0,t)':>12} | {'KPZ global':>12} | {'EW h(0,t)':>12} | {'EW global':>12}")
print("-" * 70)

for t in measure_times:
    kpz_sp = skew(kpz_h0_samples[t])
    kpz_gl = np.mean(kpz_global_skew[t])
    ew_sp = skew(ew_h0_samples[t])
    ew_gl = np.mean(ew_global_skew[t])
    
    print(f"{t:>6} | {kpz_sp:>+12.4f} | {kpz_gl:>+12.4f} | {ew_sp:>+12.4f} | {ew_gl:>+12.4f}")

# Check variance scaling
print()
print("="*78)
print("VARIANCE SCALING CHECK")
print("="*78)
print()
print("Theory: Var[h(0,t)] ~ t^(2/3) for KPZ, t^(1/2) for EW")
print()
print(f"{'Time':>6} | {'KPZ Var':>12} | {'KPZ Var/t^(2/3)':>16} | {'EW Var':>12} | {'EW Var/t^(1/2)':>16}")
print("-" * 78)

for t in measure_times:
    kpz_var = np.var(kpz_h0_samples[t])
    ew_var = np.var(ew_h0_samples[t])
    
    # Normalized by expected scaling
    kpz_norm = kpz_var / (t**(2/3))
    ew_norm = ew_var / (t**(1/2))
    
    print(f"{t:>6} | {kpz_var:>12.2f} | {kpz_norm:>16.4f} | {ew_var:>12.2f} | {ew_norm:>16.4f}")

print()
print("If properly scaled, the normalized values should be roughly constant.")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Single-point distributions at T=5000
ax1 = axes[0, 0]
t_final = 5000
kpz_data = np.array(kpz_h0_samples[t_final])
ew_data = np.array(ew_h0_samples[t_final])

# Standardize
kpz_std = (kpz_data - np.mean(kpz_data)) / np.std(kpz_data)
ew_std = (ew_data - np.mean(ew_data)) / np.std(ew_data)

ax1.hist(kpz_std, bins=30, alpha=0.6, density=True, label=f'KPZ (skew={skew(kpz_std):.3f})')
ax1.hist(ew_std, bins=30, alpha=0.6, density=True, label=f'EW (skew={skew(ew_std):.3f})')
ax1.axvline(-0.29, color='red', linestyle='--', label='TW-GUE skew=-0.29')
ax1.set_xlabel('Standardized h(0,t)')
ax1.set_ylabel('Density')
ax1.set_title(f'Single-point distributions at T={t_final}')
ax1.legend()

# Plot 2: Skewness evolution
ax2 = axes[0, 1]
kpz_skews = [skew(kpz_h0_samples[t]) for t in measure_times]
ew_skews = [skew(ew_h0_samples[t]) for t in measure_times]

ax2.plot(measure_times, kpz_skews, 'b-o', label='KPZ h(0,t)')
ax2.plot(measure_times, ew_skews, 'g-s', label='EW h(0,t)')
ax2.axhline(-0.29, color='red', linestyle='--', label='TW-GUE = -0.29')
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('Skewness')
ax2.set_title('Single-point skewness evolution')
ax2.legend()

# Plot 3: Variance scaling
ax3 = axes[1, 0]
kpz_vars = [np.var(kpz_h0_samples[t]) for t in measure_times]
ew_vars = [np.var(ew_h0_samples[t]) for t in measure_times]

ax3.loglog(measure_times, kpz_vars, 'b-o', label='KPZ')
ax3.loglog(measure_times, ew_vars, 'g-s', label='EW')

# Theory lines
t_arr = np.array(measure_times)
ax3.loglog(t_arr, 0.1 * t_arr**(2/3), 'b--', alpha=0.5, label=r'$t^{2/3}$ (KPZ theory)')
ax3.loglog(t_arr, 0.5 * t_arr**(1/2), 'g--', alpha=0.5, label=r'$t^{1/2}$ (EW theory)')

ax3.set_xlabel('Time')
ax3.set_ylabel('Var[h(0,t)]')
ax3.set_title('Variance scaling')
ax3.legend()

# Plot 4: Global vs single-point skewness
ax4 = axes[1, 1]
kpz_global = [np.mean(kpz_global_skew[t]) for t in measure_times]
ew_global = [np.mean(ew_global_skew[t]) for t in measure_times]

ax4.plot(measure_times, kpz_skews, 'b-o', label='KPZ single-point')
ax4.plot(measure_times, kpz_global, 'b--s', alpha=0.5, label='KPZ global')
ax4.plot(measure_times, ew_skews, 'g-o', label='EW single-point')
ax4.plot(measure_times, ew_global, 'g--s', alpha=0.5, label='EW global')
ax4.axhline(-0.29, color='red', linestyle='--', label='TW-GUE')
ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Time')
ax4.set_ylabel('Skewness')
ax4.set_title('Single-point vs Global skewness')
ax4.legend()

plt.tight_layout()
plt.savefig(output_dir / 'tracy_widom_proper.png', dpi=150)
print(f"\nSaved: {output_dir / 'tracy_widom_proper.png'}")

# Final verdict
print()
print("="*78)
print("VERDICT")
print("="*78)

final_kpz_skew = skew(kpz_h0_samples[5000])
if abs(final_kpz_skew - (-0.29)) < 0.1:
    print(f"✓ SUCCESS: KPZ single-point skewness = {final_kpz_skew:.3f} ≈ -0.29 (TW-GUE)")
elif final_kpz_skew < -0.1:
    print(f"~ PARTIAL: KPZ single-point skewness = {final_kpz_skew:.3f} (negative, but not -0.29)")
else:
    print(f"✗ FAILURE: KPZ single-point skewness = {final_kpz_skew:.3f} (expected -0.29)")
    print()
    print("Possible issues:")
    print("  1. Need more samples (currently 200)")
    print("  2. Need longer times (Tracy-Widom is asymptotic)")
    print("  3. Discretization effects in our Euler-Maruyama scheme")
    print("  4. Finite-size effects (L=512 may be too small)")
    print("  5. The observable we're tracking isn't quite right")

# Save summary
with open(output_dir / 'summary.txt', 'w') as f:
    f.write("Experiment 33: Proper Tracy-Widom Test\n")
    f.write("="*50 + "\n\n")
    f.write(f"Final KPZ single-point skewness: {final_kpz_skew:.4f}\n")
    f.write(f"Final EW single-point skewness: {skew(ew_h0_samples[5000]):.4f}\n")
    f.write(f"Theory (TW-GUE): -0.29\n")
    f.write(f"\nSkewness evolution:\n")
    for t in measure_times:
        f.write(f"  T={t}: KPZ={skew(kpz_h0_samples[t]):.4f}, EW={skew(ew_h0_samples[t]):.4f}\n")

print(f"Saved: {output_dir / 'summary.txt'}")
