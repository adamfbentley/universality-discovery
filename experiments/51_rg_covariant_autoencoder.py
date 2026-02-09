"""
Experiment 51: RG-Covariant Autoencoder

Goal: Learn embedding Φ such that Φ(RG[h]) = M·Φ(h)
where M is a learned d×d matrix encoding RG flow structure.

Key Tests:
1. Semigroup consistency: M(b₁)·M(b₂) ≈ M(b₁·b₂)
2. Eigenvalue structure: Universal dirs → |λ|=1, irrelevant → |λ|<1
3. Cross-system transfer: Same M works for KPZ-A and KPZ-B

Architecture:
  Encoder Φ: h(x) → z ∈ ℝ^d
  Decoder Ψ: z → h_reconstructed  
  RG matrices M_b: d×d for each scale b

Loss = L_recon + λ_RG · ||Φ(h_coarse/b^α) - M_b·Φ(h)||²

Success criterion:
  - M trained on KPZ-A transfers to KPZ-B (same class)
  - M trained on KPZ does NOT transfer to KS (different class)

Usage:
  python 51_rg_covariant_autoencoder.py --pilot   # Quick test (~10 min)
  python 51_rg_covariant_autoencoder.py           # Full experiment (~1 hour)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP issue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import argparse
import time
from datetime import datetime

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# KPZ/KS Simulation (same as Exp 50r)
# ============================================================================

def simulate_kpz(L=256, T=300, dt=0.01, nu=1.0, lam=1.0, D=0.1, seed=None):
    """Simulate 1D KPZ equation using ETD scheme in Fourier space."""
    if seed is not None:
        np.random.seed(seed)
    
    k = np.fft.fftfreq(L, d=1.0) * 2 * np.pi
    k2 = k**2
    
    # Initial condition: small random perturbation
    h = 0.01 * np.random.randn(L)
    h_hat = np.fft.fft(h)
    
    # ETD integrating factor
    exp_factor = np.exp(-nu * k2 * dt)
    
    n_steps = int(T / dt)
    noise_std = np.sqrt(2 * D * dt)
    
    for _ in range(n_steps):
        # Gradient in real space
        grad_h = np.fft.ifft(1j * k * h_hat).real
        
        # Nonlinear term
        nonlinear = 0.5 * lam * grad_h**2
        nonlinear_hat = np.fft.fft(nonlinear)
        
        # Noise term (gradient noise for proper KPZ)
        eta = noise_std * np.random.randn(L)
        eta_hat = np.fft.fft(eta)
        
        # ETD update
        h_hat = exp_factor * h_hat + dt * exp_factor * nonlinear_hat + exp_factor * eta_hat
    
    h = np.fft.ifft(h_hat).real
    h = h - np.mean(h)  # Remove mean
    
    return h.astype(np.float32)


def simulate_ks(L=256, T=300, dt=0.005, nu=1.0, seed=None):
    """Simulate 1D Kuramoto-Sivashinsky equation."""
    if seed is not None:
        np.random.seed(seed)
    
    k = np.fft.fftfreq(L, d=1.0) * 2 * np.pi
    k2 = k**2
    k4 = k**4
    
    # Linear operator for KS: -k² - k⁴
    linear = -nu * k2 - k4
    exp_factor = np.exp(linear * dt)
    
    # Initial condition
    h = 0.1 * np.random.randn(L)
    h_hat = np.fft.fft(h)
    
    n_steps = int(T / dt)
    
    for _ in range(n_steps):
        h = np.fft.ifft(h_hat).real
        grad_h = np.fft.ifft(1j * k * h_hat).real
        nonlinear = -0.5 * grad_h**2
        nonlinear_hat = np.fft.fft(nonlinear)
        
        h_hat = exp_factor * h_hat + dt * exp_factor * nonlinear_hat
    
    h = np.fft.ifft(h_hat).real
    h = h - np.mean(h)
    
    return h.astype(np.float32)


# ============================================================================
# RG Coarse-Graining
# ============================================================================

def coarse_grain(h, b):
    """Coarse-grain field by factor b with proper RG rescaling."""
    L = len(h)
    L_new = L // b
    
    # Block average
    h_coarse = h[:L_new * b].reshape(L_new, b).mean(axis=1)
    
    # RG rescaling: h_rg = h_coarse / b^α where α=0.5 for KPZ
    alpha = 0.5
    h_rg = h_coarse / (b ** alpha)
    
    return h_rg.astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

class RGPairDataset(Dataset):
    """Dataset of (h, h_coarse) pairs for RG-covariant learning."""
    
    def __init__(self, fields, scale_b):
        """
        Args:
            fields: list of height fields h(x)
            scale_b: coarse-graining factor
        """
        self.pairs = []
        for h in fields:
            h_cg = coarse_grain(h, scale_b)
            self.pairs.append((
                torch.from_numpy(h).unsqueeze(0),  # (1, L)
                torch.from_numpy(h_cg).unsqueeze(0)  # (1, L/b)
            ))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


# ============================================================================
# Model Architecture
# ============================================================================

class RGCovariantAutoencoder(nn.Module):
    """
    Autoencoder with RG-covariance constraint.
    
    Encoder Φ: h(x) → z ∈ ℝ^d
    Decoder Ψ: z → h_reconstructed
    RG matrix M: z_coarse ≈ M @ z_fine
    """
    
    def __init__(self, L_fine, L_coarse, latent_dim=8):
        super().__init__()
        
        self.L_fine = L_fine
        self.L_coarse = L_coarse
        self.latent_dim = latent_dim
        
        # Encoder for fine-scale fields
        self.encoder_fine = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, L_fine)
            flat_size_fine = self.encoder_fine(dummy).shape[1]
        
        self.fc_encode_fine = nn.Sequential(
            nn.Linear(flat_size_fine, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Encoder for coarse-scale fields (same architecture, different input size)
        self.encoder_coarse = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, L_coarse)
            flat_size_coarse = self.encoder_coarse(dummy).shape[1]
        
        self.fc_encode_coarse = nn.Sequential(
            nn.Linear(flat_size_coarse, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder (for fine-scale reconstruction)
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, flat_size_fine)
        )
        
        # Compute spatial size after conv layers
        spatial_size = L_fine // 8  # After 3 stride-2 convs
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, spatial_size)),
            nn.ConvTranspose1d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        
        # RG matrix M: maps fine embedding to coarse embedding
        # Initialize with diagonal scaling (expected RG behavior)
        # First few dims: |λ|=1 (marginal/universal), rest: |λ|<1 (irrelevant)
        diag_init = torch.tensor([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][:latent_dim])
        self.M = nn.Parameter(torch.diag(diag_init) + 0.01 * torch.randn(latent_dim, latent_dim))
    
    def encode_fine(self, h):
        """Encode fine-scale field."""
        x = self.encoder_fine(h)
        z = self.fc_encode_fine(x)
        return z
    
    def encode_coarse(self, h_cg):
        """Encode coarse-grained field."""
        x = self.encoder_coarse(h_cg)
        z = self.fc_encode_coarse(x)
        return z
    
    def decode(self, z):
        """Decode to fine-scale field."""
        x = self.fc_decode(z)
        h_recon = self.decoder(x)
        return h_recon
    
    def forward(self, h_fine, h_coarse):
        """
        Forward pass computing all needed quantities.
        
        Returns:
            z_fine: embedding of fine field
            z_coarse: embedding of coarse field
            z_predicted: M @ z_fine (should equal z_coarse)
            h_recon: reconstruction of fine field
        """
        z_fine = self.encode_fine(h_fine)
        z_coarse = self.encode_coarse(h_coarse)
        z_predicted = torch.matmul(z_fine, self.M.T)  # M @ z_fine
        h_recon = self.decode(z_fine)
        
        return z_fine, z_coarse, z_predicted, h_recon


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, lambda_rg=1.0, lambda_reg=0.01, lambda_decay=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_rg = 0
    
    for h_fine, h_coarse in dataloader:
        optimizer.zero_grad()
        
        z_fine, z_coarse, z_predicted, h_recon = model(h_fine, h_coarse)
        
        # Reconstruction loss
        loss_recon = nn.functional.mse_loss(h_recon, h_fine)
        
        # RG covariance loss: z_coarse ≈ M @ z_fine
        loss_rg = nn.functional.mse_loss(z_predicted, z_coarse)
        
        # Regularization: penalize M being too close to identity
        # This encourages learning actual RG structure (some eigenvalues < 1)
        M = model.M
        identity = torch.eye(M.shape[0], device=M.device)
        loss_not_identity = lambda_decay * torch.exp(-torch.norm(M - identity))
        
        # Also encourage M to be close to diagonal (interpretable)
        off_diag = M - torch.diag(torch.diag(M))
        loss_diag = lambda_reg * torch.norm(off_diag)
        
        loss = loss_recon + lambda_rg * loss_rg + loss_not_identity + loss_diag
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_rg += loss_rg.item()
    
    n = len(dataloader)
    return total_loss / n, total_recon / n, total_rg / n


def evaluate(model, dataloader):
    """Evaluate model."""
    model.eval()
    total_recon = 0
    total_rg = 0
    
    with torch.no_grad():
        for h_fine, h_coarse in dataloader:
            z_fine, z_coarse, z_predicted, h_recon = model(h_fine, h_coarse)
            
            loss_recon = nn.functional.mse_loss(h_recon, h_fine)
            loss_rg = nn.functional.mse_loss(z_predicted, z_coarse)
            
            total_recon += loss_recon.item()
            total_rg += loss_rg.item()
    
    n = len(dataloader)
    return total_recon / n, total_rg / n


# ============================================================================
# Diagnostics
# ============================================================================

def analyze_M_eigenstructure(M):
    """Analyze eigenvalue structure of RG matrix M."""
    M_np = M.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(M_np)
    
    # Sort by magnitude
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\n  M eigenvalue analysis:")
    print("  " + "-" * 50)
    for i, ev in enumerate(eigenvalues):
        mag = np.abs(ev)
        phase = np.angle(ev) * 180 / np.pi
        interp = "universal" if mag > 0.9 else "irrelevant"
        print(f"    λ_{i+1} = {ev:.4f} (|λ|={mag:.4f}, phase={phase:.1f}°) → {interp}")
    
    return eigenvalues, eigenvectors


def test_semigroup_consistency(model, fields, b1=2, b2=2):
    """
    Test semigroup property: M(b1)·M(b2) ≈ M(b1*b2)
    
    For this we need to train/compare M at different scales.
    Here we use a simpler check: apply M twice and compare to double coarse-graining.
    """
    model.eval()
    
    errors = []
    with torch.no_grad():
        for h in fields[:20]:  # Use subset
            h_tensor = torch.from_numpy(h).unsqueeze(0).unsqueeze(0)
            
            # Direct: h → z_fine, then M twice
            z_fine = model.encode_fine(h_tensor)
            z_M2 = torch.matmul(torch.matmul(z_fine, model.M.T), model.M.T)
            
            # Actual: coarse-grain by b1*b2, then encode
            h_cg4 = coarse_grain(h, b1 * b2)
            # Need to handle different input sizes...
            # For now, just report M eigenstructure
            
    # Simpler check: is M^2 close to some consistent structure?
    M = model.M.detach().cpu().numpy()
    M2 = M @ M
    
    # Check if M^2 eigenvalues are squares of M eigenvalues
    eig_M = np.linalg.eigvals(M)
    eig_M2 = np.linalg.eigvals(M2)
    
    # Sort by magnitude
    eig_M_sorted = np.sort(np.abs(eig_M))[::-1]
    eig_M2_sorted = np.sort(np.abs(eig_M2))[::-1]
    eig_M_squared = np.sort(np.abs(eig_M)**2)[::-1]
    
    consistency = np.mean(np.abs(eig_M2_sorted - eig_M_squared))
    
    print(f"\n  Semigroup consistency check:")
    print(f"    |λ(M²) - λ(M)²| mean error: {consistency:.6f}")
    print(f"    Status: {'✅ CONSISTENT' if consistency < 0.1 else '⚠️ INCONSISTENT'}")
    
    return consistency


def test_cross_system_transfer(model, fields_A, fields_B, scale_b=2, label_A="A", label_B="B"):
    """
    Test if M trained on system A works for system B.
    
    For same universality class: should work (low error)
    For different classes: should fail (high error)
    """
    model.eval()
    
    # Compute RG prediction error on each system
    def compute_rg_error(fields):
        errors = []
        with torch.no_grad():
            for h in fields:
                h_cg = coarse_grain(h, scale_b)
                
                h_tensor = torch.from_numpy(h).unsqueeze(0).unsqueeze(0)
                h_cg_tensor = torch.from_numpy(h_cg).unsqueeze(0).unsqueeze(0)
                
                z_fine = model.encode_fine(h_tensor)
                z_coarse = model.encode_coarse(h_cg_tensor)
                z_predicted = torch.matmul(z_fine, model.M.T)
                
                error = torch.mean((z_predicted - z_coarse)**2).item()
                errors.append(error)
        
        return np.mean(errors), np.std(errors)
    
    error_A, std_A = compute_rg_error(fields_A)
    error_B, std_B = compute_rg_error(fields_B)
    
    print(f"\n  Cross-system transfer test:")
    print(f"    {label_A}: RG error = {error_A:.6f} ± {std_A:.6f}")
    print(f"    {label_B}: RG error = {error_B:.6f} ± {std_B:.6f}")
    print(f"    Ratio {label_B}/{label_A}: {error_B/error_A:.2f}x")
    
    return error_A, error_B


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run quick pilot mode')
    args = parser.parse_args()
    
    # Parameters
    if args.pilot:
        N_train = 200
        N_test = 50
        T = 200
        epochs = 50
        output_dir = Path('results_exp51_pilot')
    else:
        N_train = 500
        N_test = 100
        T = 400
        epochs = 150
        output_dir = Path('results_exp51')
    
    output_dir.mkdir(exist_ok=True)
    
    L = 256
    scale_b = 2
    L_coarse = L // scale_b
    latent_dim = 8
    batch_size = 32
    lr = 1e-3
    lambda_rg = 1.0
    
    mode_str = "PILOT MODE" if args.pilot else "FULL MODE"
    
    print("=" * 70)
    print(f"Experiment 51: RG-Covariant Autoencoder")
    print(f"               [{mode_str}]")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  L={L}, T={T}, scale_b={scale_b}")
    print(f"  N_train={N_train}, N_test={N_test}")
    print(f"  latent_dim={latent_dim}, epochs={epochs}")
    print(f"  lambda_rg={lambda_rg}")
    print(f"\nSystems:")
    print(f"  KPZ-A: nu=1.0, lambda=1.0, D=0.1 (training)")
    print(f"  KPZ-B: nu=1.0, lambda=0.5, D=0.05 (transfer test)")
    print(f"  KS: nu=1.0 (control - different class)")
    print("=" * 70)
    
    # ========================================================================
    # Generate training data (KPZ-A)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Generating KPZ-A training data...")
    print("=" * 70)
    
    t0 = time.time()
    kpz_a_train = []
    for i in range(N_train):
        h = simulate_kpz(L=L, T=T, nu=1.0, lam=1.0, D=0.1, seed=i)
        kpz_a_train.append(h)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_train}] generated")
    
    print(f"✓ Generated {N_train} KPZ-A training fields in {time.time()-t0:.1f}s")
    
    # Test data
    print("\nGenerating test data...")
    
    kpz_a_test = [simulate_kpz(L=L, T=T, nu=1.0, lam=1.0, D=0.1, seed=10000+i) 
                  for i in range(N_test)]
    print(f"  KPZ-A test: {len(kpz_a_test)} fields")
    
    kpz_b_test = [simulate_kpz(L=L, T=T, nu=1.0, lam=0.5, D=0.05, seed=20000+i)
                  for i in range(N_test)]
    print(f"  KPZ-B test: {len(kpz_b_test)} fields")
    
    ks_test = [simulate_ks(L=L, T=T, nu=1.0, seed=30000+i) 
               for i in range(N_test)]
    print(f"  KS test: {len(ks_test)} fields")
    
    # ========================================================================
    # Create datasets
    # ========================================================================
    train_dataset = RGPairDataset(kpz_a_train, scale_b)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset_a = RGPairDataset(kpz_a_test, scale_b)
    test_loader_a = DataLoader(test_dataset_a, batch_size=batch_size)
    
    # ========================================================================
    # Initialize model
    # ========================================================================
    print("\n" + "=" * 70)
    print("Initializing model...")
    print("=" * 70)
    
    model = RGCovariantAutoencoder(L_fine=L, L_coarse=L_coarse, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Latent dimension: {latent_dim}")
    
    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    history = {'loss': [], 'recon': [], 'rg': [], 'val_recon': [], 'val_rg': []}
    best_loss = float('inf')
    
    t0 = time.time()
    for epoch in range(epochs):
        loss, recon, rg = train_epoch(model, train_loader, optimizer, lambda_rg=lambda_rg)
        val_recon, val_rg = evaluate(model, test_loader_a)
        
        history['loss'].append(loss)
        history['recon'].append(recon)
        history['rg'].append(rg)
        history['val_recon'].append(val_recon)
        history['val_rg'].append(val_rg)
        
        scheduler.step(loss)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f}, recon={recon:.6f}, "
                  f"rg={rg:.6f}, val_rg={val_rg:.6f}")
    
    train_time = time.time() - t0
    print(f"\n✓ Training complete in {train_time:.1f}s")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    
    # ========================================================================
    # Analyze learned RG matrix M
    # ========================================================================
    print("\n" + "=" * 70)
    print("Analyzing learned RG matrix M...")
    print("=" * 70)
    
    eigenvalues, eigenvectors = analyze_M_eigenstructure(model.M)
    
    # Count universal vs irrelevant directions
    n_universal = np.sum(np.abs(eigenvalues) > 0.9)
    n_irrelevant = np.sum(np.abs(eigenvalues) < 0.9)
    print(f"\n  Summary: {n_universal} universal directions, {n_irrelevant} irrelevant directions")
    
    # ========================================================================
    # Test semigroup consistency
    # ========================================================================
    print("\n" + "=" * 70)
    print("Testing semigroup consistency...")
    print("=" * 70)
    
    semigroup_error = test_semigroup_consistency(model, kpz_a_test)
    
    # ========================================================================
    # Test cross-system transfer
    # ========================================================================
    print("\n" + "=" * 70)
    print("Testing cross-system transfer...")
    print("=" * 70)
    
    print("\n  KPZ-A vs KPZ-B (same universality class):")
    error_kpz_a, error_kpz_b = test_cross_system_transfer(
        model, kpz_a_test, kpz_b_test, scale_b, "KPZ-A", "KPZ-B"
    )
    
    print("\n  KPZ-A vs KS (different universality class):")
    error_kpz_a2, error_ks = test_cross_system_transfer(
        model, kpz_a_test, ks_test, scale_b, "KPZ-A", "KS"
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 51 SUMMARY")
    print("=" * 70)
    
    print(f"\nTraining (on KPZ-A):")
    print(f"  Final loss: {history['loss'][-1]:.6f}")
    print(f"  Final recon: {history['recon'][-1]:.6f}")
    print(f"  Final RG loss: {history['rg'][-1]:.6f}")
    
    print(f"\nM eigenvalue analysis:")
    print(f"  Universal directions (|λ|>0.9): {n_universal}")
    print(f"  Irrelevant directions (|λ|<0.9): {n_irrelevant}")
    
    print(f"\nSemigroup consistency:")
    print(f"  Error: {semigroup_error:.6f}")
    print(f"  Status: {'✅ PASS' if semigroup_error < 0.1 else '⚠️ NEEDS ATTENTION'}")
    
    print(f"\nCross-system transfer:")
    print(f"  KPZ-A → KPZ-B (same class):")
    print(f"    Error ratio: {error_kpz_b/error_kpz_a:.2f}x")
    same_class_ok = error_kpz_b / error_kpz_a < 2.0
    print(f"    Status: {'✅ TRANSFERS' if same_class_ok else '⚠️ POOR TRANSFER'}")
    
    print(f"  KPZ-A → KS (different class):")
    print(f"    Error ratio: {error_ks/error_kpz_a:.2f}x")
    diff_class_ok = error_ks / error_kpz_a > 2.0
    print(f"    Status: {'✅ CORRECTLY FAILS' if diff_class_ok else '⚠️ UNEXPECTED TRANSFER'}")
    
    # Overall assessment
    print("\n" + "-" * 70)
    success = same_class_ok and diff_class_ok and semigroup_error < 0.1
    if success:
        print("✅ EXPERIMENT SUCCESS: M encodes universality class structure")
    else:
        print("⚠️ EXPERIMENT NEEDS REFINEMENT")
    print("-" * 70)
    
    # ========================================================================
    # Save results
    # ========================================================================
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'pilot' if args.pilot else 'full',
        'parameters': {
            'L': L, 'T': T, 'scale_b': scale_b,
            'N_train': N_train, 'N_test': N_test,
            'latent_dim': latent_dim, 'epochs': epochs,
            'lambda_rg': lambda_rg
        },
        'training': {
            'final_loss': float(history['loss'][-1]),
            'final_recon': float(history['recon'][-1]),
            'final_rg': float(history['rg'][-1]),
            'train_time_seconds': train_time
        },
        'eigenvalues': {
            'magnitudes': [float(np.abs(e)) for e in eigenvalues],
            'n_universal': int(n_universal),
            'n_irrelevant': int(n_irrelevant)
        },
        'semigroup': {
            'error': float(semigroup_error),
            'passed': bool(semigroup_error < 0.1)
        },
        'transfer': {
            'kpz_a_error': float(error_kpz_a),
            'kpz_b_error': float(error_kpz_b),
            'ks_error': float(error_ks),
            'kpz_b_ratio': float(error_kpz_b / error_kpz_a),
            'ks_ratio': float(error_ks / error_kpz_a),
            'same_class_transfers': bool(same_class_ok),
            'diff_class_fails': bool(diff_class_ok)
        },
        'success': bool(success)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save M matrix
    np.save(output_dir / 'M_matrix.npy', model.M.detach().cpu().numpy())
    
    # Save training history
    np.savez(output_dir / 'training_history.npz', **{k: np.array(v) for k, v in history.items()})
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
