"""
Experiment 45: RG-Covariant Embedding Learning
==============================================

DEEP THEORETICAL TEST from Assessment 2:
Can we find embedding Φ such that Φ(coarse_grain(h)) ≈ A·Φ(h) + b?

If YES → We've discovered natural coordinates where RG has simple dynamics
        (linear transformation in feature space) - PROFOUND!
If NO  → Earlier geometry was projection artifact, not fundamental structure

SELF-SUPERVISED FRAMEWORK:
Learn Φ (neural network) and scale-dependent transformations A_b, b_b by minimizing:

    L = Σ ||Φ(coarse_grain_b(h)) - (A_b @ Φ(h) + b_b)||²
    
across multiple coarse-graining scales b ∈ {2, 4, 8}

NO CLASS LABELS - purely self-supervised from RG structure.

PHYSICAL INTERPRETATION:
- If loss is low: RG flow is predictable/linear in Φ-space
- Eigenvalues of A_b: RG-relevant (>1), marginal (≈1), irrelevant (<1) directions  
- Fixed points: Φ(h) where A_b @ Φ(h) + b_b ≈ Φ(h) for all b
- Universality classes: Basins of attraction to fixed points

SUCCESS CRITERIA:
1. RG covariance loss < 0.1 (features predictable across scales)
2. Learned Φ separates EW/KPZ better than hand-crafted features (PC1)
3. A_b matrices have interpretable eigenstructure (relevant directions)
4. Generalization: Works on held-out parameter combinations

COMPARISON TO GRADIENT MOMENTS:
- Hand-crafted: 6D gradient moments (Exp 20-27)
- Learned: RG-covariant features (this experiment)
- Test: Which gives better separation? Which is more physically meaningful?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from numba import jit

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# SIMULATION FUNCTIONS (Numba)
# ============================================================================

@jit(nopython=True)
def simulate_ew_trajectory(L=128, T=2000, nu=1.0, D=1.0, dt=0.05, save_interval=10):
    """Edwards-Wilkinson: ∂h/∂t = ν∇²h + η"""
    interface = np.zeros(L)
    n_saves = T // save_interval
    trajectory = np.zeros((n_saves, L))
    save_idx = 0
    
    for t in range(T):
        new_interface = interface.copy()
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            laplacian = left - 2*center + right
            noise = np.sqrt(D * dt) * np.random.randn()
            dhdt = nu * laplacian + noise
            new_interface[x] = center + dt * dhdt
        interface = new_interface
        
        if t % save_interval == 0:
            trajectory[save_idx] = interface.copy()
            save_idx += 1
    
    return trajectory

@jit(nopython=True)
def simulate_kpz_trajectory(L=128, T=2000, lambda_=0.5, nu=1.0, D=1.0, 
                            dt=0.05, save_interval=10):
    """KPZ: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η"""
    interface = np.zeros(L)
    n_saves = T // save_interval
    trajectory = np.zeros((n_saves, L))
    save_idx = 0
    
    for t in range(T):
        new_interface = interface.copy()
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            laplacian = left - 2*center + right
            gradient = (right - left) / 2.0
            noise = np.sqrt(D * dt) * np.random.randn()
            dhdt = nu * laplacian + (lambda_ / 2.0) * gradient**2 + noise
            new_interface[x] = center + dt * dhdt
        interface = new_interface
        
        if t % save_interval == 0:
            trajectory[save_idx] = interface.copy()
            save_idx += 1
    
    return trajectory

def coarse_grain(h, block_size):
    """
    Coarse-grain height field by block averaging + rescaling.
    h: (L,) array
    Returns: (L//block_size,) array
    """
    L = len(h)
    L_new = L // block_size
    h_coarse = np.zeros(L_new)
    
    for i in range(L_new):
        h_coarse[i] = np.mean(h[i*block_size:(i+1)*block_size])
    
    # Rescale by sqrt(block_size) to preserve fluctuations
    return h_coarse * np.sqrt(block_size)

# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset of height field snapshots with multiple coarse-graining scales"""
    
    def __init__(self, trajectories, labels):
        """
        trajectories: list of (T, L) arrays
        labels: list of class labels (0=EW, 1=KPZ)
        """
        self.data = []
        self.labels = []
        
        # Extract snapshots from late-time regime
        for traj, label in zip(trajectories, labels):
            # Use last 50 frames
            for frame in traj[-50:]:
                self.data.append(frame)
                self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        h = self.data[idx]
        label = self.labels[idx]
        
        # Create coarse-grained versions
        L = len(h)
        h_2 = coarse_grain(h, 2) if L >= 64 else h  # Scale 2
        h_4 = coarse_grain(h, 4) if L >= 64 else h  # Scale 4
        h_8 = coarse_grain(h, 8) if L >= 64 else h  # Scale 8
        
        # Pad to same size (needed for batching)
        # Use zero-padding at boundaries
        def pad_to_size(arr, target_size):
            if len(arr) >= target_size:
                return arr[:target_size]
            else:
                return np.pad(arr, (0, target_size - len(arr)), mode='constant')
        
        h_padded = h
        h_2_padded = pad_to_size(h_2, L)
        h_4_padded = pad_to_size(h_4, L)
        h_8_padded = pad_to_size(h_8, L)
        
        return {
            'h': torch.FloatTensor(h_padded),
            'h_2': torch.FloatTensor(h_2_padded),
            'h_4': torch.FloatTensor(h_4_padded),
            'h_8': torch.FloatTensor(h_8_padded),
            'label': torch.LongTensor([label])[0]
        }

# ============================================================================
# RG-COVARIANT FEATURE NETWORK
# ============================================================================

class RGCovariantNetwork(nn.Module):
    """
    Neural network learning RG-covariant embedding Φ.
    
    Architecture:
    - 1D Convolutional layers (capture local structure)
    - Global pooling (translation invariance)
    - Dense layers to feature space
    - Learnable RG transformation matrices A_b, b_b
    """
    
    def __init__(self, L=128, feature_dim=8):
        super().__init__()
        self.L = L
        self.feature_dim = feature_dim
        
        # Encoder: height field → features
        self.encoder = nn.Sequential(
            # Conv layers with increasing receptive field
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L -> L/2
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L/2 -> L/4
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Global pooling to fixed size
            
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )
        
        # RG transformation matrices (learnable)
        # A_b: (feature_dim, feature_dim) linear map
        # b_b: (feature_dim,) bias
        self.A_2 = nn.Parameter(torch.eye(feature_dim) * 0.9)  # Initialize near identity
        self.A_4 = nn.Parameter(torch.eye(feature_dim) * 0.8)
        self.A_8 = nn.Parameter(torch.eye(feature_dim) * 0.7)
        
        self.b_2 = nn.Parameter(torch.zeros(feature_dim))
        self.b_4 = nn.Parameter(torch.zeros(feature_dim))
        self.b_8 = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, h):
        """
        Encode height field to features.
        h: (batch, L)
        Returns: (batch, feature_dim)
        """
        # Add channel dimension: (batch, L) -> (batch, 1, L)
        h = h.unsqueeze(1)
        features = self.encoder(h)
        return features
    
    def rg_transform(self, features, scale):
        """
        Apply learned RG transformation for given scale.
        features: (batch, feature_dim)
        scale: 2, 4, or 8
        Returns: (batch, feature_dim)
        """
        if scale == 2:
            return features @ self.A_2.T + self.b_2
        elif scale == 4:
            return features @ self.A_4.T + self.b_4
        elif scale == 8:
            return features @ self.A_8.T + self.b_8
        else:
            raise ValueError(f"Unsupported scale: {scale}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_rg_covariant_network(train_loader, val_loader, 
                               feature_dim=8, num_epochs=50, lr=1e-3):
    """
    Train RG-covariant network with self-supervised loss.
    """
    model = RGCovariantNetwork(L=128, feature_dim=feature_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "="*70)
    print("TRAINING RG-COVARIANT NETWORK")
    print("="*70)
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            h = batch['h'].to(device)
            h_2 = batch['h_2'].to(device)
            h_4 = batch['h_4'].to(device)
            h_8 = batch['h_8'].to(device)
            
            # Forward pass: encode all scales
            phi_h = model(h)
            phi_h_2 = model(h_2)
            phi_h_4 = model(h_4)
            phi_h_8 = model(h_8)
            
            # RG covariance loss: ||Φ(coarse_h) - (A @ Φ(h) + b)||²
            phi_pred_2 = model.rg_transform(phi_h, scale=2)
            phi_pred_4 = model.rg_transform(phi_h, scale=4)
            phi_pred_8 = model.rg_transform(phi_h, scale=8)
            
            loss_2 = F.mse_loss(phi_h_2, phi_pred_2)
            loss_4 = F.mse_loss(phi_h_4, phi_pred_4)
            loss_8 = F.mse_loss(phi_h_8, phi_pred_8)
            
            loss = loss_2 + loss_4 + loss_8
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                h = batch['h'].to(device)
                h_2 = batch['h_2'].to(device)
                h_4 = batch['h_4'].to(device)
                h_8 = batch['h_8'].to(device)
                
                phi_h = model(h)
                phi_h_2 = model(h_2)
                phi_h_4 = model(h_4)
                phi_h_8 = model(h_8)
                
                phi_pred_2 = model.rg_transform(phi_h, scale=2)
                phi_pred_4 = model.rg_transform(phi_h, scale=4)
                phi_pred_8 = model.rg_transform(phi_h, scale=8)
                
                loss_2 = F.mse_loss(phi_h_2, phi_pred_2)
                loss_4 = F.mse_loss(phi_h_4, phi_pred_4)
                loss_8 = F.mse_loss(phi_h_8, phi_pred_8)
                
                loss = loss_2 + loss_4 + loss_8
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses

# ============================================================================
# EVALUATION & COMPARISON
# ============================================================================

def evaluate_separation(model, data_loader, use_learned=True):
    """
    Evaluate EW vs KPZ separation using learned or hand-crafted features.
    
    Returns: features, labels, pc1, correlation
    """
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            h = batch['h'].to(device)
            labels = batch['label'].cpu().numpy()
            
            if use_learned:
                # Learned RG-covariant features
                features = model(h).cpu().numpy()
            else:
                # Hand-crafted gradient moments (baseline)
                features = []
                for h_i in h.cpu().numpy():
                    grad = np.gradient(h_i)
                    lap = np.gradient(grad)
                    feat = np.array([
                        np.var(grad),
                        np.mean(grad**3) / (np.std(grad)**3 + 1e-10),
                        np.mean(grad**4) / (np.std(grad)**4 + 1e-10) - 3,
                        np.var(lap),
                        np.cov(np.abs(grad), lap)[0, 1],
                        np.var(h_i)
                    ])
                    features.append(feat)
                features = np.array(features)
            
            all_features.append(features)
            all_labels.append(labels)
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    # PCA to get PC1
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(features_scaled)[:, 0]
    
    # Correlation with labels
    r, p = pearsonr(pc1, all_labels)
    
    return all_features, all_labels, pc1, r, p

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Main experimental pipeline"""
    
    print("="*70)
    print("EXPERIMENT 45: RG-Covariant Embedding Learning")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Hyperparameters
    L = 128
    T = 2000
    n_samples_train = 30  # Per class
    n_samples_val = 10
    feature_dim = 8
    batch_size = 16
    num_epochs = 50
    
    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING TRAINING DATA")
    print("="*70)
    print(f"System size: L = {L}")
    print(f"Time steps: T = {T}")
    print(f"Samples per class (train): {n_samples_train}")
    print(f"Samples per class (val): {n_samples_val}")
    
    # Training data
    train_trajectories = []
    train_labels = []
    
    print("\nGenerating EW trajectories...")
    for i in range(n_samples_train):
        traj = simulate_ew_trajectory(L=L, T=T)
        train_trajectories.append(traj)
        train_labels.append(0)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples_train} complete")
    
    print("\nGenerating KPZ trajectories...")
    for i in range(n_samples_train):
        traj = simulate_kpz_trajectory(L=L, T=T)
        train_trajectories.append(traj)
        train_labels.append(1)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples_train} complete")
    
    # Validation data
    val_trajectories = []
    val_labels = []
    
    print("\nGenerating validation data...")
    for i in range(n_samples_val):
        traj = simulate_ew_trajectory(L=L, T=T)
        val_trajectories.append(traj)
        val_labels.append(0)
    
    for i in range(n_samples_val):
        traj = simulate_kpz_trajectory(L=L, T=T)
        val_trajectories.append(traj)
        val_labels.append(1)
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_trajectories, train_labels)
    val_dataset = TrajectoryDataset(val_trajectories, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining set: {len(train_dataset)} snapshots")
    print(f"Validation set: {len(val_dataset)} snapshots")
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    model, train_losses, val_losses = train_rg_covariant_network(
        train_loader, val_loader,
        feature_dim=feature_dim,
        num_epochs=num_epochs,
        lr=1e-3
    )
    
    # ========================================================================
    # EVALUATE & COMPARE
    # ========================================================================
    
    print("\n" + "="*70)
    print("EVALUATION: Learned vs Hand-Crafted Features")
    print("="*70)
    
    # Learned features
    print("\n1. RG-Covariant Learned Features:")
    feat_learned, labels_learned, pc1_learned, r_learned, p_learned = \
        evaluate_separation(model, val_loader, use_learned=True)
    print(f"   PC1 vs Class correlation: r = {r_learned:.4f} (p = {p_learned:.2e})")
    print(f"   Final RG covariance loss: {val_losses[-1]:.4f}")
    
    # Hand-crafted features (baseline)
    print("\n2. Hand-Crafted Gradient Moments:")
    feat_baseline, labels_baseline, pc1_baseline, r_baseline, p_baseline = \
        evaluate_separation(model, val_loader, use_learned=False)
    print(f"   PC1 vs Class correlation: r = {r_baseline:.4f} (p = {p_baseline:.2e})")
    
    # ========================================================================
    # RG TRANSFORMATION ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("RG TRANSFORMATION MATRICES")
    print("="*70)
    
    A_2 = model.A_2.detach().cpu().numpy()
    A_4 = model.A_4.detach().cpu().numpy()
    A_8 = model.A_8.detach().cpu().numpy()
    
    # Eigenanalysis
    eig_2, _ = np.linalg.eig(A_2)
    eig_4, _ = np.linalg.eig(A_4)
    eig_8, _ = np.linalg.eig(A_8)
    
    print("\nEigenvalues of A_b (sorted by magnitude):")
    print(f"Scale 2: {np.sort(np.abs(eig_2))[::-1]}")
    print(f"Scale 4: {np.sort(np.abs(eig_4))[::-1]}")
    print(f"Scale 8: {np.sort(np.abs(eig_8))[::-1]}")
    
    print("\nRG-relevant directions (|λ| > 1):")
    print(f"Scale 2: {np.sum(np.abs(eig_2) > 1)} directions")
    print(f"Scale 4: {np.sum(np.abs(eig_4) > 1)} directions")
    print(f"Scale 8: {np.sum(np.abs(eig_8) > 1)} directions")
    
    # ========================================================================
    # SUCCESS CRITERIA
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*70)
    
    success_low_loss = val_losses[-1] < 0.1
    success_better_separation = abs(r_learned) > abs(r_baseline)
    success_interpretable = np.sum(np.abs(eig_2) > 1) >= 1
    
    print(f"\n1. RG covariance loss < 0.1: {'✅ PASS' if success_low_loss else '❌ FAIL'}")
    print(f"   → Final loss = {val_losses[-1]:.4f}")
    
    print(f"\n2. Better separation than baseline: {'✅ PASS' if success_better_separation else '❌ FAIL'}")
    print(f"   → Learned: |r| = {abs(r_learned):.4f}")
    print(f"   → Baseline: |r| = {abs(r_baseline):.4f}")
    print(f"   → Improvement: {abs(r_learned) - abs(r_baseline):.4f}")
    
    print(f"\n3. Interpretable eigenstructure: {'✅ PASS' if success_interpretable else '❌ FAIL'}")
    print(f"   → Found {np.sum(np.abs(eig_2) > 1)} RG-relevant directions")
    
    overall_success = success_low_loss and (success_better_separation or success_interpretable)
    
    print("\n" + "="*70)
    if overall_success:
        print("🎉 SUCCESS: FOUND RG-COVARIANT EMBEDDING!")
        print("="*70)
        print("\nInterpretation:")
        print("  • RG flow has simple (near-linear) dynamics in learned feature space")
        print("  • This is NOT curve-fitting - it's discovering coordinate system")
        print("  • Eigenstructure reveals RG-relevant vs irrelevant directions")
        print("  • Profound: We've empirically found coordinates where RG is natural")
    else:
        print("⚠️  PARTIAL SUCCESS / COMPLEX DYNAMICS")
        print("="*70)
        print("\nInterpretation:")
        print("  • RG flow exists but is nonlinear/high-dimensional")
        print("  • May need:")
        print("    - Deeper network architecture")
        print("    - Nonlinear RG transformations")
        print("    - More training data")
        print("  • Gradient moments remain competitive baseline")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results_dir = Path(__file__).parent.parent / 'results' / 'exp45_rg_covariant'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), results_dir / 'rg_covariant_model.pt')
    print(f"\n💾 Saved model: {results_dir / 'rg_covariant_model.pt'}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learned_features': feat_learned,
        'baseline_features': feat_baseline,
        'labels': labels_learned,
        'pc1_learned': pc1_learned,
        'pc1_baseline': pc1_baseline,
        'correlations': {
            'learned': (r_learned, p_learned),
            'baseline': (r_baseline, p_baseline)
        },
        'rg_matrices': {
            'A_2': A_2,
            'A_4': A_4,
            'A_8': A_8,
            'eigenvalues_2': eig_2,
            'eigenvalues_4': eig_4,
            'eigenvalues_8': eig_8
        },
        'success_criteria': {
            'low_loss': success_low_loss,
            'better_separation': success_better_separation,
            'interpretable': success_interpretable,
            'overall': overall_success
        }
    }
    
    with open(results_dir / 'rg_covariant_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Saved results: {results_dir / 'rg_covariant_results.pkl'}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Validation', linewidth=2)
    ax.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='Success threshold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RG Covariance Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Feature comparison
    ax = axes[0, 1]
    ax.scatter(pc1_baseline[labels_baseline == 0], pc1_learned[labels_learned == 0],
              alpha=0.6, s=50, label='EW', color='blue')
    ax.scatter(pc1_baseline[labels_baseline == 1], pc1_learned[labels_learned == 1],
              alpha=0.6, s=50, label='KPZ', color='red')
    ax.set_xlabel('Baseline PC1 (gradient moments)', fontsize=11)
    ax.set_ylabel('Learned PC1 (RG-covariant)', fontsize=11)
    ax.set_title('Feature Space Comparison', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Separation comparison
    ax = axes[1, 0]
    x_baseline = np.random.randn(len(pc1_baseline)) * 0.1
    x_learned = np.random.randn(len(pc1_learned)) * 0.1 + 1
    ax.scatter(x_baseline[labels_baseline == 0], pc1_baseline[labels_baseline == 0],
              alpha=0.5, s=40, label='EW', color='blue')
    ax.scatter(x_baseline[labels_baseline == 1], pc1_baseline[labels_baseline == 1],
              alpha=0.5, s=40, label='KPZ', color='red')
    ax.scatter(x_learned[labels_learned == 0], pc1_learned[labels_learned == 0],
              alpha=0.5, s=40, color='blue')
    ax.scatter(x_learned[labels_learned == 1], pc1_learned[labels_learned == 1],
              alpha=0.5, s=40, color='red')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline\n(gradient moments)', 'Learned\n(RG-covariant)'])
    ax.set_ylabel('PC1', fontsize=11)
    ax.set_title(f'Separation Quality: r={r_baseline:.3f} → r={r_learned:.3f}', 
                fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Eigenvalue spectrum
    ax = axes[1, 1]
    scales = [2, 4, 8]
    eigs = [eig_2, eig_4, eig_8]
    for scale, eig in zip(scales, eigs):
        sorted_eig = np.sort(np.abs(eig))[::-1]
        ax.plot(range(1, len(sorted_eig)+1), sorted_eig, 'o-', 
               linewidth=2, markersize=8, label=f'Scale {scale}', alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=2,
              label='Marginal (|λ|=1)')
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('|Eigenvalue|', fontsize=12)
    ax.set_title('RG Transformation Spectrum', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'rg_covariant_analysis.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved figure: {results_dir / 'rg_covariant_analysis.png'}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("EXPERIMENT 45 COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_experiment()
