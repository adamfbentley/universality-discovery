"""
Validation Tests for Exp 50i - Feature Degeneracy Check

CRITICAL: Exp 50i showed -52% distance drop, but feature ranges suggest degeneracy.

This script runs 4 validation tests:
1. KPZ vs KPZ with spectral features (should stay flat)
2. Feature evolution plots (check if collapse vs convergence)
3. Rerun MMD without degenerate features
4. Cross-check with mean distance (not MMD)

If convergence persists: Real signal ✅
If it disappears: Feature collapse artifact ❌
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import linregress

sns.set_style("whitegrid")

# Load 50i results
results_path = Path('results/exp50i_diagnostic_b_spectral/results.pkl')
with open(results_path, 'rb') as f:
    exp50i = pickle.load(f)

print("="*70)
print("VALIDATION TESTS FOR EXP 50i")
print("="*70)
print("\nGoal: Check if -52% convergence is real or feature degeneracy artifact")
print("="*70)

# Load features
ks_fields_path = Path('results/exp50i_diagnostic_b_spectral/ks_fields.pkl')
kpz_fields_path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')

with open(ks_fields_path, 'rb') as f:
    ks_data = pickle.load(f)

with open(kpz_fields_path, 'rb') as f:
    kpz_data = pickle.load(f)

# Need to recompute features with debugging
from scipy.fft import fft, fftfreq

def compute_power_spectrum(h):
    """Compute power spectrum."""
    L = len(h)
    h_fft = fft(h)
    S = np.abs(h_fft)**2 / L
    k = fftfreq(L, d=1.0)
    positive_k_mask = k > 0
    k_pos = k[positive_k_mask]
    S_pos = S[positive_k_mask]
    return k_pos, S_pos

def extract_spectral_shape_features_debug(h, scale_name=''):
    """Extract with debugging info."""
    k, S = compute_power_spectrum(h)
    
    total_power = np.sum(S)
    if total_power < 1e-12:
        return np.zeros(5), {}
    
    k_max = np.max(k)
    
    # Band definition (this is the problem!)
    k_low_cutoff = 0.2 * k_max
    k_mid_cutoff = 0.5 * k_max
    
    low_mask = k < k_low_cutoff
    mid_mask = (k >= k_low_cutoff) & (k < k_mid_cutoff)
    high_mask = k >= k_mid_cutoff
    
    f_low = np.sum(S[low_mask]) / total_power if np.any(low_mask) else 0.0
    f_mid = np.sum(S[mid_mask]) / total_power if np.any(mid_mask) else 0.0
    f_high = np.sum(S[high_mask]) / total_power if np.any(high_mask) else 0.0
    
    # Low-k slope
    fit_mask = (k > 0) & (k < 0.3 * k_max)
    if np.sum(fit_mask) >= 5:
        k_fit = k[fit_mask]
        S_fit = S[fit_mask]
        valid = S_fit > 1e-12
        if np.sum(valid) >= 3:
            log_k = np.log(k_fit[valid])
            log_S = np.log(S_fit[valid])
            slope, _, _, _, _ = linregress(log_k, log_S)
        else:
            slope = 0.0
    else:
        slope = 0.0
    
    k_cent = np.sum(k * S) / total_power
    k_cent_norm = k_cent / k_max
    
    debug_info = {
        'k_max': k_max,
        'k_low_cutoff': k_low_cutoff,
        'n_low': np.sum(low_mask),
        'n_mid': np.sum(mid_mask),
        'n_high': np.sum(high_mask),
        'n_total': len(k)
    }
    
    features = np.array([slope, f_low, f_mid, f_high, k_cent_norm])
    return features, debug_info

# Check feature degeneracy at each scale
print("\n" + "="*70)
print("TEST 0: FEATURE DEGENERACY DIAGNOSIS")
print("="*70)

scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
scale_names = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']

from scipy.fft import ifft as scipyifft

def coarse_grain_field_spectral(h, k_cutoff_fraction):
    """Spectral low-pass."""
    L = len(h)
    h_hat = fft(h)
    k = np.fft.fftfreq(L)
    k_max = np.max(np.abs(k))
    k_cutoff = k_cutoff_fraction * k_max
    mask = np.abs(k) <= k_cutoff
    h_hat_filtered = h_hat * mask
    return np.real(scipyifft(h_hat_filtered))

for scale, name in zip(scales, scale_names):
    print(f"\n{name} (k_cutoff_fraction = {scale}):")
    
    # Sample one KS field
    h_ks_raw = ks_data['fields'][0]
    h_ks = coarse_grain_field_spectral(h_ks_raw, scale)
    features_ks, debug_ks = extract_spectral_shape_features_debug(h_ks, name)
    
    # Sample one KPZ field
    h_kpz_raw = kpz_data['fields'][0]
    h_kpz = coarse_grain_field_spectral(h_kpz_raw, scale)
    features_kpz, debug_kpz = extract_spectral_shape_features_debug(h_kpz, name)
    
    print(f"  KS debug: k_max={debug_ks['k_max']:.4f}, k_low_cutoff={debug_ks['k_low_cutoff']:.4f}")
    print(f"           n_modes: low={debug_ks['n_low']}, mid={debug_ks['n_mid']}, high={debug_ks['n_high']}, total={debug_ks['n_total']}")
    print(f"           Features: {features_ks}")
    
    print(f"  KPZ debug: k_max={debug_kpz['k_max']:.4f}, k_low_cutoff={debug_kpz['k_low_cutoff']:.4f}")
    print(f"            n_modes: low={debug_kpz['n_low']}, mid={debug_kpz['n_mid']}, high={debug_kpz['n_high']}, total={debug_kpz['n_total']}")
    print(f"            Features: {features_kpz}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("If n_low ≈ n_total at large b: ALL modes in 'low' band → f_low=1 (degenerate!)")
print("This is FEATURE COLLAPSE, not physical convergence.")
print("="*70)
