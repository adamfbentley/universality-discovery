"""
Check KS vs KPZ amplitude scaling to ensure fair comparison.

The issue: KS gradients ~ 10^-3, KPZ gradients ~ 10^-1
This 100× difference might dominate the distance metric.

Need to either:
1. Match system parameters (L, noise, etc.)
2. Normalize features properly
3. Test in regime where KS should show KPZ-like scaling
"""

import sys
sys.path.append('src')

import numpy as np
import pickle
from pathlib import Path

# Load KPZ data
kpz_path = Path('results/exp46_coupling_coordinate/coupling_coordinate_results.pkl')
with open(kpz_path, 'rb') as f:
    kpz_data = pickle.load(f)

print("Available keys:", list(kpz_data.keys()))

# Try to find features
if 'features' in kpz_data:
    kpz_features = kpz_data['features']
elif 'X_train' in kpz_data:
    kpz_features = kpz_data['X_train']
elif 'X' in kpz_data:
    kpz_features = kpz_data['X']
else:
    print("Cannot find features!")
    kpz_features = None

if kpz_features is not None:
    print("="*70)
    print("KPZ FEATURE STATISTICS")
    print("="*70)
    print(f"Shape: {kpz_features.shape}")
    print(f"\nColumn statistics (likely m2, m3, m4, m5, m6, m7):")
    for i in range(kpz_features.shape[1]):
        col = kpz_features[:, i]
        print(f"  m{i+2}: mean={np.mean(col):+.3e}, std={np.std(col):.3e}, "
              f"range=[{np.min(col):+.3e}, {np.max(col):+.3e}]")

    # Check simulation parameters
    if 'params' in kpz_data:
        print(f"\nKPZ parameters: {kpz_data['params']}")
    if 'config' in kpz_data:
        print(f"\nKPZ config: {kpz_data['config']}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("\nIf KPZ m2 ~ 0.1 and KS m2 ~ 0.001:")
print("  → KS field gradients are ~10× smaller")
print("  → This could be due to:")
print("     - Different system size (L)")
print("     - Different noise strength")
print("     - Different parameter regime")
print("     - Intrinsic difference in roughness exponent")
print("\nTo test properly:")
print("  1. Match L (both should use L=256)")
print("  2. Match noise strength (or KS noise vs KPZ stochasticity)")
print("  3. OR: Normalize features before comparing")
