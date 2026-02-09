"""
Experiment 36: Proper Droplet IC for Tracy-Widom GUE

The Tracy-Widom GUE distribution appears for KPZ with "narrow wedge" IC:
  h(x,0) = -|x|/ε  as ε → 0  (delta function limit)

This represents growth starting from a POINT SOURCE (single seed).

The key is that the initial "droplet" should be LOCALIZED, not spanning
the whole domain. The interface then grows outward from this point.

For flat IC → TW-GOE (we verified this works: skewness ≈ +0.22)
For narrow wedge IC → TW-GUE (skewness ≈ -0.29)

Reference: Sasamoto & Spohn (2010), Calabrese & Le Doussal (2011)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
import time

np.random.seed(42)

output_dir = Path(__file__).parent.parent / 'results' / 'exp36_proper_droplet'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*78)
print("EXPERIMENT 36: PROPER DROPLET IC FOR TRACY-WIDOM GUE")
print("="*78)
print()

# Parameters optimized from Exp35
L = 4096
T = 500
dt = 0.01
n_samples = 500

print(f"Parameters: L={L}, T={T}, dt={dt}, n_samples={n_samples}")
print()

# The KEY insight: For TW-GUE, we need a NARROW WEDGE initial condition
# h(x,0) = -|x - x0| / ε  for small ε
# This creates a sharp peak at x0

# Different wedge widths to test
wedge_widths = [1, 5, 10, 50]  # ε values

results = {}

for width in wedge_widths:
    print(f"Testing wedge width ε = {width}...")
    
    kpz_heights = []
    ew_heights = []
    
    start = time.time()
    for s in range(n_samples):
        if (s+1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  Sample {s+1}/{n_samples} ({elapsed:.1f}s)")
        
        center = L // 2
        
        # NARROW WEDGE IC: h(x,0) = -|x - center| / width
        # But we cap it to avoid huge negative values
        h_kpz = np.zeros(L)
        h_ew = np.zeros(L)
        
        for x in range(L):
            dist = min(abs(x - center), L - abs(x - center))
            # Narrow wedge: sharp peak at center, decaying to -max_depth
            if dist <= width * 10:  # Only affect region near center
                h_kpz[x] = -dist / width
                h_ew[x] = -dist / width
            else:
                h_kpz[x] = -10  # Flat floor
                h_ew[x] = -10
        
        # Track height at the peak (center)
        h0_kpz = h_kpz[center]
        h0_ew = h_ew[center]
        
        # Simulate
        n_steps = int(T / dt)
        for step in range(n_steps):
            # EW
            lap_ew = np.roll(h_ew, 1) + np.roll(h_ew, -1) - 2*h_ew
            h_ew = h_ew + dt * lap_ew + np.sqrt(dt) * np.random.randn(L)
            
            # KPZ
            lap_kpz = np.roll(h_kpz, 1) + np.roll(h_kpz, -1) - 2*h_kpz
            grad_kpz = (np.roll(h_kpz, -1) - np.roll(h_kpz, 1)) / 2
            h_kpz = h_kpz + dt * (lap_kpz + 0.5 * grad_kpz**2) + np.sqrt(dt) * np.random.randn(L)
        
        kpz_heights.append(h_kpz[center] - h0_kpz)
        ew_heights.append(h_ew[center] - h0_ew)
    
    # Analyze
    kpz_arr = np.array(kpz_heights)
    ew_arr = np.array(ew_heights)
    
    kpz_chi = (kpz_arr - np.mean(kpz_arr)) / np.std(kpz_arr)
    ew_chi = (ew_arr - np.mean(ew_arr)) / np.std(ew_arr)
    
    kpz_skew = skew(kpz_chi)
    ew_skew = skew(ew_chi)
    
    results[width] = {
        'kpz_skew': kpz_skew,
        'ew_skew': ew_skew,
        'kpz_kurt': kurtosis(kpz_chi, fisher=True),
        'kpz_data': kpz_chi,
        'ew_data': ew_chi
    }
    
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  KPZ skewness: {kpz_skew:+.4f} (theory: -0.29)")
    print(f"  EW skewness:  {ew_skew:+.4f} (theory: 0)")
    print()

# Summary
print("="*78)
print("RESULTS SUMMARY: WEDGE WIDTH DEPENDENCE")
print("="*78)
print()
print(f"{'Width ε':>10} | {'KPZ skewness':>14} | {'EW skewness':>14} | {'Match TW-GUE?':>14}")
print("-" * 60)

for width in wedge_widths:
    r = results[width]
    match = "✓" if abs(r['kpz_skew'] - (-0.29)) < 0.1 else "✗"
    print(f"{width:>10} | {r['kpz_skew']:>+14.4f} | {r['ew_skew']:>+14.4f} | {match:>14}")

print()
print("Theory: TW-GUE skewness = -0.2935")
print("        TW-GOE skewness = +0.2935 (flat IC)")
print()

# Check if narrower wedge helps
best_width = min(wedge_widths, key=lambda w: abs(results[w]['kpz_skew'] - (-0.29)))
print(f"Best match: ε = {best_width}, KPZ skewness = {results[best_width]['kpz_skew']:+.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Skewness vs wedge width
ax1 = axes[0, 0]
widths = list(results.keys())
kpz_skews = [results[w]['kpz_skew'] for w in widths]
ew_skews = [results[w]['ew_skew'] for w in widths]

ax1.plot(widths, kpz_skews, 'bo-', markersize=10, label='KPZ')
ax1.plot(widths, ew_skews, 'gs-', markersize=10, label='EW')
ax1.axhline(-0.29, color='red', linestyle='--', label='TW-GUE = -0.29')
ax1.axhline(+0.29, color='orange', linestyle='--', label='TW-GOE = +0.29')
ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Wedge width ε')
ax1.set_ylabel('Skewness')
ax1.set_title('Skewness vs Initial Condition Width')
ax1.legend()
ax1.set_xscale('log')

# Plot 2: Distribution for narrowest wedge
ax2 = axes[0, 1]
w = wedge_widths[0]  # Narrowest
x_range = np.linspace(-4, 4, 100)
ax2.hist(results[w]['kpz_data'], bins=40, density=True, alpha=0.6, 
         label=f'KPZ (skew={results[w]["kpz_skew"]:.3f})')
ax2.hist(results[w]['ew_data'], bins=40, density=True, alpha=0.6,
         label=f'EW (skew={results[w]["ew_skew"]:.3f})')
from scipy.stats import norm
ax2.plot(x_range, norm.pdf(x_range), 'k--', label='Gaussian', alpha=0.7)
ax2.axvline(-0.29, color='red', linestyle=':', linewidth=2)
ax2.set_xlabel('χ = (h - ⟨h⟩) / σ')
ax2.set_ylabel('Density')
ax2.set_title(f'Narrow Wedge IC (ε={w})')
ax2.legend()

# Plot 3: Distribution for widest wedge
ax3 = axes[1, 0]
w = wedge_widths[-1]  # Widest
ax3.hist(results[w]['kpz_data'], bins=40, density=True, alpha=0.6,
         label=f'KPZ (skew={results[w]["kpz_skew"]:.3f})')
ax3.hist(results[w]['ew_data'], bins=40, density=True, alpha=0.6,
         label=f'EW (skew={results[w]["ew_skew"]:.3f})')
ax3.plot(x_range, norm.pdf(x_range), 'k--', label='Gaussian', alpha=0.7)
ax3.set_xlabel('χ = (h - ⟨h⟩) / σ')
ax3.set_ylabel('Density')
ax3.set_title(f'Wide Wedge IC (ε={w})')
ax3.legend()

# Plot 4: Initial conditions visualization
ax4 = axes[1, 1]
x = np.arange(200) - 100  # Show region around center
for w in wedge_widths:
    h_init = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if abs(xi) <= w * 10:
            h_init[i] = -abs(xi) / w
        else:
            h_init[i] = -10
    ax4.plot(x, h_init, label=f'ε = {w}')
ax4.set_xlabel('x - center')
ax4.set_ylabel('h(x, 0)')
ax4.set_title('Initial Condition Profiles')
ax4.legend()
ax4.set_ylim(-15, 1)

plt.tight_layout()
plt.savefig(output_dir / 'proper_droplet.png', dpi=150)
print(f"Saved: {output_dir / 'proper_droplet.png'}")

# Final verdict
print()
print("="*78)
print("VERDICT")
print("="*78)

narrowest = wedge_widths[0]
if abs(results[narrowest]['kpz_skew'] - (-0.29)) < 0.1:
    print("✓ SUCCESS: Narrow wedge IC produces TW-GUE statistics!")
    print(f"  Narrowest wedge (ε={narrowest}): skewness = {results[narrowest]['kpz_skew']:+.4f}")
elif results[narrowest]['kpz_skew'] < 0:
    print("~ PARTIAL: Negative skewness observed (correct sign for TW-GUE)")
    print(f"  But magnitude differs: {results[narrowest]['kpz_skew']:+.4f} vs -0.29")
    print()
    print("Possible improvements:")
    print("  1. Even narrower wedge (ε < 1)")
    print("  2. More samples for better statistics")
    print("  3. Different observable (max height vs center)")
else:
    print("✗ FAILURE: Still getting positive skewness")
    print()
    print("This suggests the observable needs to change, not just the IC.")
    print("Consider: tracking the MAXIMUM height, not the center point")

# Save summary
with open(output_dir / 'summary.txt', 'w') as f:
    f.write("Experiment 36: Proper Droplet IC\n")
    f.write("="*50 + "\n\n")
    for w in wedge_widths:
        f.write(f"Wedge width ε={w}: KPZ skew = {results[w]['kpz_skew']:+.4f}\n")
    f.write(f"\nBest: ε={best_width}, skew = {results[best_width]['kpz_skew']:+.4f}\n")
    f.write(f"Theory (TW-GUE): -0.29\n")

print(f"Saved: {output_dir / 'summary.txt'}")
