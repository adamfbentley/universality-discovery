"""
Experiment 35: Correct KPZ Observable with Literature Parameters

The problem: We've been computing statistics WRONG.

For KPZ height fluctuations with FLAT IC:
  h(x,t) = v∞ t + (Γt)^(1/3) χ + O(1)

where χ follows Tracy-Widom GOE (not GUE!) for flat IC.
TW-GOE has skewness ≈ +0.2935 (positive!)

For DROPLET (curved) IC:
  χ follows Tracy-Widom GUE with skewness ≈ -0.2935 (negative)

The CORRECT observable:
  χ = [h(x,t) - <h(x,t)>] / σ(t)
  where σ(t) ~ t^(1/3)

Let's use parameters from exact solution literature:
  Sasamoto & Spohn (2010), Amir, Corwin & Quastel (2011)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis, norm
import time

np.random.seed(123)

output_dir = Path(__file__).parent.parent / 'results' / 'exp35_correct_observable'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*78)
print("EXPERIMENT 35: CORRECT KPZ OBSERVABLE")
print("="*78)
print()

# Key insight: We need to RESCALE properly
# For flat IC: 
#   h(x,t) fluctuations scale as t^(1/3)
#   Width scales as t^(1/2) (roughness)
#   
# The Tracy-Widom distribution describes the CENTERED, RESCALED height
# NOT the raw statistics of h(x)

# Parameters: Use canonical values
L = 4096  # Very large to avoid finite-size
T = 500   # Not too long so L >> √T
dt = 0.01  # Fine timestep for accuracy
n_samples = 500  # Many samples for statistics

print(f"Parameters: L={L}, T={T}, dt={dt}, n_samples={n_samples}")
print(f"L/√T = {L/np.sqrt(T):.1f} >> 1 (avoids finite-size effects)")
print()

# We'll track h at the center point
# For FLAT IC, use TW-GOE reference (skewness ≈ +0.29)
# For DROPLET IC, use TW-GUE reference (skewness ≈ -0.29)

print("="*78)
print("PART 1: FLAT IC (Expect TW-GOE, skewness ≈ +0.29)")
print("="*78)
print()

kpz_h_center_flat = []
ew_h_center_flat = []

start = time.time()
for s in range(n_samples):
    if (s+1) % 50 == 0:
        elapsed = time.time() - start
        print(f"  Sample {s+1}/{n_samples} ({elapsed:.1f}s)")
    
    # FLAT initial condition: h(x,0) = 0 + small noise
    h_kpz = np.zeros(L) 
    h_ew = np.zeros(L)
    
    # Store initial height at center
    center = L // 2
    
    # Simulate with SMALLER dt for stability
    n_steps = int(T / dt)
    for step in range(n_steps):
        # EW dynamics
        lap_ew = np.roll(h_ew, 1) + np.roll(h_ew, -1) - 2*h_ew
        h_ew = h_ew + dt * lap_ew + np.sqrt(dt) * np.random.randn(L)
        
        # KPZ dynamics
        lap_kpz = np.roll(h_kpz, 1) + np.roll(h_kpz, -1) - 2*h_kpz
        grad_kpz = (np.roll(h_kpz, -1) - np.roll(h_kpz, 1)) / 2
        h_kpz = h_kpz + dt * (lap_kpz + 0.5 * grad_kpz**2) + np.sqrt(dt) * np.random.randn(L)
    
    kpz_h_center_flat.append(h_kpz[center])
    ew_h_center_flat.append(h_ew[center])

print(f"\nCompleted in {time.time() - start:.1f}s")

# Analyze: center and rescale
kpz_flat = np.array(kpz_h_center_flat)
ew_flat = np.array(ew_h_center_flat)

# The fluctuation χ = (h - <h>) / σ
kpz_chi_flat = (kpz_flat - np.mean(kpz_flat)) / np.std(kpz_flat)
ew_chi_flat = (ew_flat - np.mean(ew_flat)) / np.std(ew_flat)

print()
print("FLAT IC Results:")
print(f"  KPZ: mean={np.mean(kpz_flat):.2f}, std={np.std(kpz_flat):.2f}")
print(f"       skewness={skew(kpz_chi_flat):.4f} (theory: +0.29 for TW-GOE)")
print(f"       kurtosis={kurtosis(kpz_chi_flat, fisher=True):.4f}")
print(f"  EW:  mean={np.mean(ew_flat):.2f}, std={np.std(ew_flat):.2f}")
print(f"       skewness={skew(ew_chi_flat):.4f} (theory: 0 for Gaussian)")
print()

# Variance scaling check
print("Variance scaling check:")
print(f"  KPZ std = {np.std(kpz_flat):.2f}")
print(f"  Expected ~ T^(1/3) = {T**(1/3):.2f}")
print(f"  Ratio = {np.std(kpz_flat) / T**(1/3):.3f}")
print()

print("="*78)
print("PART 2: DROPLET IC (Expect TW-GUE, skewness ≈ -0.29)")
print("="*78)
print()

kpz_h_center_drop = []
ew_h_center_drop = []

start = time.time()
for s in range(n_samples):
    if (s+1) % 50 == 0:
        elapsed = time.time() - start
        print(f"  Sample {s+1}/{n_samples} ({elapsed:.1f}s)")
    
    # DROPLET initial condition: h(x,0) = -|x - L/2| (wedge/parabola)
    center = L // 2
    h_kpz = np.zeros(L)
    h_ew = np.zeros(L)
    
    for x in range(L):
        dist = min(abs(x - center), L - abs(x - center))  # periodic distance
        h_kpz[x] = -dist * 0.5  # Steeper droplet
        h_ew[x] = -dist * 0.5
    
    # Track the PEAK (where droplet started)
    h0_kpz = h_kpz[center]
    h0_ew = h_ew[center]
    
    # Simulate
    n_steps = int(T / dt)
    for step in range(n_steps):
        # EW dynamics
        lap_ew = np.roll(h_ew, 1) + np.roll(h_ew, -1) - 2*h_ew
        h_ew = h_ew + dt * lap_ew + np.sqrt(dt) * np.random.randn(L)
        
        # KPZ dynamics
        lap_kpz = np.roll(h_kpz, 1) + np.roll(h_kpz, -1) - 2*h_kpz
        grad_kpz = (np.roll(h_kpz, -1) - np.roll(h_kpz, 1)) / 2
        h_kpz = h_kpz + dt * (lap_kpz + 0.5 * grad_kpz**2) + np.sqrt(dt) * np.random.randn(L)
    
    # Track height relative to initial
    kpz_h_center_drop.append(h_kpz[center] - h0_kpz)
    ew_h_center_drop.append(h_ew[center] - h0_ew)

print(f"\nCompleted in {time.time() - start:.1f}s")

# Analyze
kpz_drop = np.array(kpz_h_center_drop)
ew_drop = np.array(ew_h_center_drop)

kpz_chi_drop = (kpz_drop - np.mean(kpz_drop)) / np.std(kpz_drop)
ew_chi_drop = (ew_drop - np.mean(ew_drop)) / np.std(ew_drop)

print()
print("DROPLET IC Results:")
print(f"  KPZ: mean={np.mean(kpz_drop):.2f}, std={np.std(kpz_drop):.2f}")
print(f"       skewness={skew(kpz_chi_drop):.4f} (theory: -0.29 for TW-GUE)")
print(f"       kurtosis={kurtosis(kpz_chi_drop, fisher=True):.4f}")
print(f"  EW:  mean={np.mean(ew_drop):.2f}, std={np.std(ew_drop):.2f}")
print(f"       skewness={skew(ew_chi_drop):.4f} (theory: 0 for Gaussian)")
print()

# Summary
print("="*78)
print("SUMMARY")
print("="*78)
print()
print("                 |   KPZ skewness   |   EW skewness")
print("                 | Observed  Theory | Observed  Theory")
print("-" * 60)
print(f" FLAT IC (GOE)   | {skew(kpz_chi_flat):+7.4f}   +0.29  | {skew(ew_chi_flat):+7.4f}    0.00")
print(f" DROPLET IC(GUE) | {skew(kpz_chi_drop):+7.4f}   -0.29  | {skew(ew_chi_drop):+7.4f}    0.00")
print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Flat IC distributions
ax1 = axes[0, 0]
x_range = np.linspace(-4, 4, 100)
ax1.hist(kpz_chi_flat, bins=40, density=True, alpha=0.6, label=f'KPZ (skew={skew(kpz_chi_flat):.3f})')
ax1.hist(ew_chi_flat, bins=40, density=True, alpha=0.6, label=f'EW (skew={skew(ew_chi_flat):.3f})')
ax1.plot(x_range, norm.pdf(x_range), 'k--', label='Gaussian', alpha=0.7)
ax1.axvline(0.29, color='red', linestyle=':', label='TW-GOE skew (+0.29)')
ax1.set_xlabel('χ = (h - ⟨h⟩) / σ')
ax1.set_ylabel('Density')
ax1.set_title('FLAT IC: Expected TW-GOE (skew ≈ +0.29)')
ax1.legend()
ax1.set_xlim(-4, 4)

# Plot 2: Droplet IC distributions
ax2 = axes[0, 1]
ax2.hist(kpz_chi_drop, bins=40, density=True, alpha=0.6, label=f'KPZ (skew={skew(kpz_chi_drop):.3f})')
ax2.hist(ew_chi_drop, bins=40, density=True, alpha=0.6, label=f'EW (skew={skew(ew_chi_drop):.3f})')
ax2.plot(x_range, norm.pdf(x_range), 'k--', label='Gaussian', alpha=0.7)
ax2.axvline(-0.29, color='red', linestyle=':', label='TW-GUE skew (-0.29)')
ax2.set_xlabel('χ = (h - ⟨h⟩) / σ')
ax2.set_ylabel('Density')
ax2.set_title('DROPLET IC: Expected TW-GUE (skew ≈ -0.29)')
ax2.legend()
ax2.set_xlim(-4, 4)

# Plot 3: Raw distributions (not rescaled)
ax3 = axes[1, 0]
ax3.hist(kpz_flat, bins=40, alpha=0.6, label=f'KPZ flat (mean={np.mean(kpz_flat):.1f})')
ax3.hist(kpz_drop, bins=40, alpha=0.6, label=f'KPZ droplet (mean={np.mean(kpz_drop):.1f})')
ax3.set_xlabel('h(x=L/2, t=T)')
ax3.set_ylabel('Count')
ax3.set_title('Raw height distributions (before rescaling)')
ax3.legend()

# Plot 4: Summary text
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
EXPERIMENT 35 RESULTS
=====================

Parameters:
  L = {L} (large to avoid finite-size effects)
  T = {T}
  n_samples = {n_samples}
  dt = {dt}

FLAT Initial Condition (TW-GOE expected):
  KPZ skewness: {skew(kpz_chi_flat):+.4f}  (theory: +0.29)
  EW skewness:  {skew(ew_chi_flat):+.4f}  (theory: 0)
  
  Status: {'✓ MATCHES' if abs(skew(kpz_chi_flat) - 0.29) < 0.1 else '✗ MISMATCH'}

DROPLET Initial Condition (TW-GUE expected):
  KPZ skewness: {skew(kpz_chi_drop):+.4f}  (theory: -0.29)
  EW skewness:  {skew(ew_chi_drop):+.4f}  (theory: 0)
  
  Status: {'✓ MATCHES' if abs(skew(kpz_chi_drop) - (-0.29)) < 0.1 else '✗ MISMATCH'}

EW should always be Gaussian (skew ≈ 0):
  Flat IC:    {'✓' if abs(skew(ew_chi_flat)) < 0.2 else '✗'}
  Droplet IC: {'✓' if abs(skew(ew_chi_drop)) < 0.2 else '✗'}
"""
ax4.text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
         verticalalignment='top', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(output_dir / 'correct_observable.png', dpi=150)
print(f"Saved: {output_dir / 'correct_observable.png'}")

# Final verdict
print()
print("="*78)
print("VERDICT")
print("="*78)

flat_kpz_ok = abs(skew(kpz_chi_flat) - 0.29) < 0.15
drop_kpz_ok = abs(skew(kpz_chi_drop) - (-0.29)) < 0.15
ew_ok = abs(skew(ew_chi_flat)) < 0.2 and abs(skew(ew_chi_drop)) < 0.2

if flat_kpz_ok and drop_kpz_ok:
    print("✓ SUCCESS: Tracy-Widom statistics observed!")
    print("  - Flat IC: KPZ shows TW-GOE (positive skewness)")
    print("  - Droplet IC: KPZ shows TW-GUE (negative skewness)")
    print("  - EW shows Gaussian (zero skewness)")
elif flat_kpz_ok or drop_kpz_ok:
    print("~ PARTIAL: Some Tracy-Widom features observed")
    print(f"  - Flat IC: {'✓' if flat_kpz_ok else '✗'}")
    print(f"  - Droplet IC: {'✓' if drop_kpz_ok else '✗'}")
else:
    print("✗ STILL NOT SEEING TRACY-WIDOM")
    print()
    print("Possible remaining issues:")
    print("  1. Need even more samples (currently 500)")
    print("  2. Numerical discretization effects")
    print("  3. Euler-Maruyama not accurate enough for higher moments")
    print("  4. The exact TW parameters depend on the precise KPZ equation form")

with open(output_dir / 'summary.txt', 'w') as f:
    f.write(f"Exp35: Correct Observable Test\n")
    f.write(f"L={L}, T={T}, n_samples={n_samples}\n\n")
    f.write(f"FLAT IC: KPZ skew = {skew(kpz_chi_flat):.4f}, EW skew = {skew(ew_chi_flat):.4f}\n")
    f.write(f"DROPLET IC: KPZ skew = {skew(kpz_chi_drop):.4f}, EW skew = {skew(ew_chi_drop):.4f}\n")
