"""
Experiment 45b: RG-Covariant Embedding with Anti-Collapse Constraints
=====================================================================

QUICK FIX for Exp 45 degenerate solution.

PROBLEM: Pure self-supervised RG loss has trivial solution Φ ≈ 0
SOLUTION: Multi-task learning with classification head as regularization

Loss = L_RG_covariance + λ * L_classification

This forces network to learn discriminative features (via classification)
while maintaining RG structure (via covariance loss).

ARCHITECTURAL CHANGES:
1. Feature normalization (unit sphere)
2. Classification head added (EW vs KPZ)
3. Joint loss with weight balance
4. Use droplet IC (known to give strong separation from Exp 27)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from numba import jit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# SIMULATION FUNCTIONS (WITH DROPLET IC)
# ============================================================================

@jit(nopython=True)
def generate_droplet_ic(L):
    """Droplet IC: h(x,0) = -|x - L/2|"""
    x = np.arange(L)
    return -np.abs(x - L/2).astype(np.float64)

@jit(nopython=True)
def simulate_ew_trajectory_droplet(L=128, T=2000, nu=1.0, D=1.0, dt=0.05, save_interval=10):
    """EW with droplet IC"""
    interface = generate_droplet_ic(L)
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
def simulate_kpz_trajectory_droplet(L=128, T=2000, lambda_=0.5, nu=1.0, D=1.0, 
                                    dt=0.05, save_interval=10):
    """KPZ with droplet IC"""
    interface = generate_droplet_ic(L)
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
    """Block averaging + rescaling"""
    L = len(h)
    L_new = L // block_size
    h_coarse = np.zeros(L_new)
    
    for i in range(L_new):
        h_coarse[i] = np.mean(h[i*block_size:(i+1)*block_size])
    
    return h_coarse * np.sqrt(block_size)

# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels):
        self.data = []
        self.labels = []
        
        for traj, label in zip(trajectories, labels):
            for frame in traj[-50:]:  # Last 50 frames
                self.data.append(frame)
                self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        h = self.data[idx]
        label = self.labels[idx]
        L = len(h)
        
        h_2 = coarse_grain(h, 2) if L >= 64 else h
        h_4 = coarse_grain(h, 4) if L >= 64 else h
        h_8 = coarse_grain(h, 8) if L >= 64 else h
        
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
# IMPROVED RG-COVARIANT NETWORK WITH ANTI-COLLAPSE
# ============================================================================

class RGCovariantNetworkV2(nn.Module):
    """
    Improved architecture with anti-collapse mechanisms:
    1. Feature normalization (prevents Φ→0)
    2. Classification head (forces discriminative features)
    3. Batch normalization (stabilizes training)
    """
    
    def __init__(self, L=128, feature_dim=8, num_classes=2):
        super().__init__()
        self.L = L
        self.feature_dim = feature_dim
        
        # Encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )
        
        # Classification head (for anti-collapse)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # RG transformation matrices
        self.A_2 = nn.Parameter(torch.eye(feature_dim) * 0.9)
        self.A_4 = nn.Parameter(torch.eye(feature_dim) * 0.8)
        self.A_8 = nn.Parameter(torch.eye(feature_dim) * 0.7)
        
        self.b_2 = nn.Parameter(torch.zeros(feature_dim))
        self.b_4 = nn.Parameter(torch.zeros(feature_dim))
        self.b_8 = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, h, normalize=True):
        """
        Encode with optional normalization.
        """
        h = h.unsqueeze(1)
        features = self.encoder(h)
        
        # CRITICAL: Normalize to unit sphere (prevents collapse to 0)
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def classify(self, features):
        """Classification logits"""
        return self.classifier(features)
    
    def rg_transform(self, features, scale):
        """Apply RG transformation"""
        if scale == 2:
            return features @ self.A_2.T + self.b_2
        elif scale == 4:
            return features @ self.A_4.T + self.b_4
        elif scale == 8:
            return features @ self.A_8.T + self.b_8
        else:
            raise ValueError(f"Unsupported scale: {scale}")

# ============================================================================
# TRAINING WITH MULTI-TASK LOSS
# ============================================================================

def train_multitask(train_loader, val_loader, feature_dim=8, num_epochs=50, 
                    lr=1e-3, lambda_class=0.1):
    """
    Train with joint loss: L_RG + λ * L_classification
    """
    model = RGCovariantNetworkV2(L=128, feature_dim=feature_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses_rg = []
    train_losses_class = []
    val_losses_rg = []
    val_losses_class = []
    
    print("\n" + "="*70)
    print("TRAINING RG-COVARIANT NETWORK V2 (MULTI-TASK)")
    print("="*70)
    print(f"Feature dimension: {feature_dim}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Classification weight λ: {lambda_class}")
    print(f"Device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_rg = 0.0
        epoch_loss_class = 0.0
        
        for batch in train_loader:
            h = batch['h'].to(device)
            h_2 = batch['h_2'].to(device)
            h_4 = batch['h_4'].to(device)
            h_8 = batch['h_8'].to(device)
            labels = batch['label'].to(device)
            
            # Forward: encode all scales (with normalization)
            phi_h = model(h, normalize=True)
            phi_h_2 = model(h_2, normalize=True)
            phi_h_4 = model(h_4, normalize=True)
            phi_h_8 = model(h_8, normalize=True)
            
            # RG covariance loss
            phi_pred_2 = model.rg_transform(phi_h, scale=2)
            phi_pred_4 = model.rg_transform(phi_h, scale=4)
            phi_pred_8 = model.rg_transform(phi_h, scale=8)
            
            loss_rg_2 = F.mse_loss(phi_h_2, phi_pred_2)
            loss_rg_4 = F.mse_loss(phi_h_4, phi_pred_4)
            loss_rg_8 = F.mse_loss(phi_h_8, phi_pred_8)
            loss_rg = loss_rg_2 + loss_rg_4 + loss_rg_8
            
            # Classification loss (anti-collapse regularizer)
            logits = model.classify(phi_h)
            loss_class = F.cross_entropy(logits, labels)
            
            # Joint loss
            loss = loss_rg + lambda_class * loss_class
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss_rg += loss_rg.item()
            epoch_loss_class += loss_class.item()
        
        avg_train_loss_rg = epoch_loss_rg / len(train_loader)
        avg_train_loss_class = epoch_loss_class / len(train_loader)
        train_losses_rg.append(avg_train_loss_rg)
        train_losses_class.append(avg_train_loss_class)
        
        # Validation
        model.eval()
        val_loss_rg = 0.0
        val_loss_class = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                h = batch['h'].to(device)
                h_2 = batch['h_2'].to(device)
                h_4 = batch['h_4'].to(device)
                h_8 = batch['h_8'].to(device)
                labels = batch['label'].to(device)
                
                phi_h = model(h, normalize=True)
                phi_h_2 = model(h_2, normalize=True)
                phi_h_4 = model(h_4, normalize=True)
                phi_h_8 = model(h_8, normalize=True)
                
                phi_pred_2 = model.rg_transform(phi_h, scale=2)
                phi_pred_4 = model.rg_transform(phi_h, scale=4)
                phi_pred_8 = model.rg_transform(phi_h, scale=8)
                
                loss_rg = (F.mse_loss(phi_h_2, phi_pred_2) + 
                          F.mse_loss(phi_h_4, phi_pred_4) + 
                          F.mse_loss(phi_h_8, phi_pred_8))
                
                logits = model.classify(phi_h)
                loss_class = F.cross_entropy(logits, labels)
                
                val_loss_rg += loss_rg.item()
                val_loss_class += loss_class.item()
        
        avg_val_loss_rg = val_loss_rg / len(val_loader)
        avg_val_loss_class = val_loss_class / len(val_loader)
        val_losses_rg.append(avg_val_loss_rg)
        val_losses_class.append(avg_val_loss_class)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"RG={avg_train_loss_rg:.4f}, Class={avg_train_loss_class:.4f} | "
                  f"Val RG={avg_val_loss_rg:.4f}, Class={avg_val_loss_class:.4f}")
    
    return model, train_losses_rg, train_losses_class, val_losses_rg, val_losses_class

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_separation(model, data_loader, use_learned=True):
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            h = batch['h'].to(device)
            labels = batch['label'].cpu().numpy()
            
            if use_learned:
                features = model(h, normalize=True).cpu().numpy()
            else:
                # Baseline gradient moments
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
    
    # PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(features_scaled)[:, 0]
    
    r, p = pearsonr(pc1, all_labels)
    
    return all_features, all_labels, pc1, r, p

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("="*70)
    print("EXPERIMENT 45b: RG-Covariant with Anti-Collapse")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Hyperparameters
    L = 128
    T = 2000
    n_samples_train = 30
    n_samples_val = 10
    feature_dim = 8
    batch_size = 16
    num_epochs = 50
    lambda_class = 0.1  # Classification weight
    
    print(f"\nKEY CHANGE: Using DROPLET IC (Exp 27 showed r=-0.98 separation)")
    
    # Generate data with droplet IC
    print("\n" + "="*70)
    print("GENERATING TRAINING DATA (DROPLET IC)")
    print("="*70)
    
    train_trajectories = []
    train_labels = []
    
    print("\nGenerating EW trajectories (droplet IC)...")
    for i in range(n_samples_train):
        traj = simulate_ew_trajectory_droplet(L=L, T=T)
        train_trajectories.append(traj)
        train_labels.append(0)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples_train} complete")
    
    print("\nGenerating KPZ trajectories (droplet IC)...")
    for i in range(n_samples_train):
        traj = simulate_kpz_trajectory_droplet(L=L, T=T)
        train_trajectories.append(traj)
        train_labels.append(1)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples_train} complete")
    
    # Validation data
    val_trajectories = []
    val_labels = []
    
    print("\nGenerating validation data (droplet IC)...")
    for i in range(n_samples_val):
        traj = simulate_ew_trajectory_droplet(L=L, T=T)
        val_trajectories.append(traj)
        val_labels.append(0)
    
    for i in range(n_samples_val):
        traj = simulate_kpz_trajectory_droplet(L=L, T=T)
        val_trajectories.append(traj)
        val_labels.append(1)
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_trajectories, train_labels)
    val_dataset = TrajectoryDataset(val_trajectories, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining set: {len(train_dataset)} snapshots")
    print(f"Validation set: {len(val_dataset)} snapshots")
    
    # Train
    model, train_rg, train_class, val_rg, val_class = train_multitask(
        train_loader, val_loader,
        feature_dim=feature_dim,
        num_epochs=num_epochs,
        lr=1e-3,
        lambda_class=lambda_class
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    print("\n1. Learned Features (normalized):")
    feat_learned, labels_learned, pc1_learned, r_learned, p_learned = \
        evaluate_separation(model, val_loader, use_learned=True)
    print(f"   PC1 vs Class: r = {r_learned:.4f} (p = {p_learned:.2e})")
    print(f"   RG covariance loss: {val_rg[-1]:.4f}")
    
    print("\n2. Baseline Gradient Moments:")
    feat_baseline, labels_baseline, pc1_baseline, r_baseline, p_baseline = \
        evaluate_separation(model, val_loader, use_learned=False)
    print(f"   PC1 vs Class: r = {r_baseline:.4f} (p = {p_baseline:.2e})")
    
    # RG matrices
    print("\n" + "="*70)
    print("RG TRANSFORMATION MATRICES")
    print("="*70)
    
    A_2 = model.A_2.detach().cpu().numpy()
    A_4 = model.A_4.detach().cpu().numpy()
    A_8 = model.A_8.detach().cpu().numpy()
    
    eig_2, _ = np.linalg.eig(A_2)
    eig_4, _ = np.linalg.eig(A_4)
    eig_8, _ = np.linalg.eig(A_8)
    
    print(f"\nEigenvalue magnitudes (scale 2): {np.sort(np.abs(eig_2))[::-1][:4]}")
    print(f"Eigenvalue magnitudes (scale 4): {np.sort(np.abs(eig_4))[::-1][:4]}")
    print(f"Eigenvalue magnitudes (scale 8): {np.sort(np.abs(eig_8))[::-1][:4]}")
    
    print(f"\nRG-relevant directions (|λ| > 1):")
    print(f"  Scale 2: {np.sum(np.abs(eig_2) > 1)}")
    print(f"  Scale 4: {np.sum(np.abs(eig_4) > 1)}")
    print(f"  Scale 8: {np.sum(np.abs(eig_8) > 1)}")
    
    # Success criteria
    print("\n" + "="*70)
    print("SUCCESS CRITERIA")
    print("="*70)
    
    success_nondeg = not np.isnan(r_learned) and abs(r_learned) > 0.3
    success_better = abs(r_learned) > abs(r_baseline)
    success_rg_reasonable = val_rg[-1] < 1.0
    
    print(f"\n1. Non-degenerate features (|r| > 0.3): {'✅ PASS' if success_nondeg else '❌ FAIL'}")
    print(f"   → |r_learned| = {abs(r_learned):.4f}")
    
    print(f"\n2. Better than baseline: {'✅ PASS' if success_better else '❌ FAIL'}")
    print(f"   → Learned: {abs(r_learned):.4f}")
    print(f"   → Baseline: {abs(r_baseline):.4f}")
    print(f"   → Improvement: {abs(r_learned) - abs(r_baseline):.4f}")
    
    print(f"\n3. Reasonable RG loss: {'✅ PASS' if success_rg_reasonable else '❌ FAIL'}")
    print(f"   → Final loss = {val_rg[-1]:.4f}")
    
    overall_success = success_nondeg and success_rg_reasonable
    
    print("\n" + "="*70)
    if overall_success:
        print("🎉 SUCCESS: NON-DEGENERATE RG-COVARIANT FEATURES!")
        print("="*70)
        print("\nInterpretation:")
        print("  • Multi-task learning prevented feature collapse")
        print("  • Learned features are discriminative AND RG-structured")
        if success_better:
            print("  • Outperforms hand-crafted gradient moments")
        print("  • Ready for deeper analysis and manuscript integration")
    else:
        print("⚠️  MIXED RESULTS")
        print("="*70)
        print("\nNeeds investigation - check training curves and feature distributions")
    
    # Save
    results_dir = Path(__file__).parent.parent / 'results' / 'exp45b_rg_covariant_v2'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / 'model_v2.pt')
    
    results = {
        'train_losses_rg': train_rg,
        'train_losses_class': train_class,
        'val_losses_rg': val_rg,
        'val_losses_class': val_class,
        'learned_features': feat_learned,
        'baseline_features': feat_baseline,
        'labels': labels_learned,
        'pc1_learned': pc1_learned,
        'pc1_baseline': pc1_baseline,
        'correlations': {
            'learned': (r_learned, p_learned),
            'baseline': (r_baseline, p_baseline)
        },
        'rg_matrices': {'A_2': A_2, 'A_4': A_4, 'A_8': A_8},
        'eigenvalues': {'eig_2': eig_2, 'eig_4': eig_4, 'eig_8': eig_8},
        'success': overall_success
    }
    
    with open(results_dir / 'results_v2.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Saved: {results_dir / 'model_v2.pt'}")
    print(f"💾 Saved: {results_dir / 'results_v2.pkl'}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.plot(train_rg, label='Train RG', linewidth=2)
    ax.plot(val_rg, label='Val RG', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RG Covariance Loss')
    ax.set_title('RG Loss Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(train_class, label='Train Class', linewidth=2)
    ax.plot(val_class, label='Val Class', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Loss')
    ax.set_title('Classification Loss Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(pc1_baseline[labels_baseline==0], pc1_learned[labels_learned==0],
              alpha=0.6, s=50, label='EW', color='blue')
    ax.scatter(pc1_baseline[labels_baseline==1], pc1_learned[labels_learned==1],
              alpha=0.6, s=50, label='KPZ', color='red')
    ax.set_xlabel('Baseline PC1')
    ax.set_ylabel('Learned PC1')
    ax.set_title('Feature Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    scales = [2, 4, 8]
    eigs = [eig_2, eig_4, eig_8]
    for scale, eig in zip(scales, eigs):
        sorted_eig = np.sort(np.abs(eig))[::-1]
        ax.plot(range(1, len(sorted_eig)+1), sorted_eig, 'o-', 
               linewidth=2, markersize=8, label=f'Scale {scale}', alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Marginal')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('|Eigenvalue|')
    ax.set_title('RG Eigenspectrum')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'analysis_v2.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {results_dir / 'analysis_v2.png'}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("EXPERIMENT 45b COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    results = run_experiment()
