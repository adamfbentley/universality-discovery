"""
Experiment 48: Domain-Adversarial IC Factorization

MOTIVATION:
-----------
Exp 45b achieved r=-1.000 separation but ONLY with droplet IC (Exp 27: r=-0.98).
Flat IC completely fails (r=-0.06). This suggests features conflate:
  (universality class, IC sector) → region
rather than pure class separation.

HYPOTHESIS:
-----------
Can we learn features that are:
1. Predictive of universality class (EW vs KPZ)
2. Invariant to initial condition type (flat vs droplet)

This is a DOMAIN ADAPTATION problem: IC type is the "domain", class is the label.

APPROACH:
---------
Domain-adversarial training (Ganin & Lempitsky 2015):
  
  Feature extractor Φ_θ: h → ℝ^d
  Class predictor C_ϕ: Φ → {EW, KPZ}
  Domain predictor D_ψ: Φ → {flat, droplet}
  
  Loss = L_class(C_ϕ(Φ_θ(h)), true_class)           [maximize]
       - λ * L_domain(D_ψ(Φ_θ(h)), true_IC)         [minimize via gradient reversal]
       
  Goal: Features that predict class but are IC-blind (domain predictor fails).

ARCHITECTURE:
-------------
  Input: gradient moments (m2, m3, m4, m5, m6, m7) [6D]
  
  Φ_θ: [6] → [16] → [16] → [8]  (feature extractor, BatchNorm, Dropout)
  C_ϕ: [8] → [4] → [2]           (class predictor)
  D_ψ: [8] → [4] → [2]           (domain predictor with GradientReversal)

EXPECTED OUTCOMES:
------------------
SUCCESS: Class accuracy ~100%, Domain accuracy ~50% (random)
         → Features encode universality, not IC
         
FAILURE: Class accuracy ~100%, Domain accuracy ~100%
         → Features still encode IC (inseparable from class)

DATASET:
--------
- EW + KPZ, both with {flat, droplet} IC
- 4 combinations: (EW,flat), (EW,droplet), (KPZ,flat), (KPZ,droplet)
- Need sufficient data from each to avoid trivial solutions

Status: NEW (Feb 3, 2026) - Following Assessment 3 three-pillar validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Reverses gradients during backprop (Ganin & Lempitsky 2015).
    Forward: identity
    Backward: -λ * grad
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

# ============================================================================
# 2. NETWORKS
# ============================================================================

class FeatureExtractor(nn.Module):
    """Φ_θ: gradient moments → feature space"""
    def __init__(self, input_dim=6, hidden_dim=16, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ClassPredictor(nn.Module):
    """C_ϕ: features → {EW, KPZ}"""
    def __init__(self, feature_dim=8, hidden_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 classes
        )
    
    def forward(self, features):
        return self.net(features)

class DomainPredictor(nn.Module):
    """D_ψ: features → {flat, droplet}"""
    def __init__(self, feature_dim=8, hidden_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 domains
        )
    
    def forward(self, features):
        return self.net(features)

class DomainAdversarialModel(nn.Module):
    """Full model with gradient reversal"""
    def __init__(self, input_dim=6, hidden_dim=16, feature_dim=8):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim)
        self.class_predictor = ClassPredictor(feature_dim)
        self.domain_predictor = DomainPredictor(feature_dim)
    
    def forward(self, x, lambda_grl=1.0):
        features = self.feature_extractor(x)
        class_logits = self.class_predictor(features)
        
        # Gradient reversal for domain predictor
        reversed_features = gradient_reversal(features, lambda_grl)
        domain_logits = self.domain_predictor(reversed_features)
        
        return features, class_logits, domain_logits

# ============================================================================
# 3. DATASET
# ============================================================================

class SurfaceGrowthDataset(Dataset):
    def __init__(self, features, class_labels, domain_labels):
        """
        features: (N, 6) array of gradient moments
        class_labels: (N,) array of {0: EW, 1: KPZ}
        domain_labels: (N,) array of {0: flat, 1: droplet}
        """
        self.features = torch.FloatTensor(features)
        self.class_labels = torch.LongTensor(class_labels)
        self.domain_labels = torch.LongTensor(domain_labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.class_labels[idx], self.domain_labels[idx]

# ============================================================================
# 4. LOAD DATA
# ============================================================================

def load_surface_growth_data():
    """
    Load gradient moment features from experiments.
    Expected structure: results/exp48_domain_adversarial/data.pkl
    
    For now, we'll generate synthetic data matching the observed patterns:
    - EW/KPZ separation via PC1
    - Droplet IC amplifies separation
    - Flat IC weakens it
    """
    print("Loading/generating data...")
    
    # Check if real data exists
    data_path = Path('results/exp48_domain_adversarial/data.pkl')
    if data_path.exists():
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # Generate synthetic data matching observed patterns
    np.random.seed(42)
    n_samples = 500  # per combination
    
    # Base feature generation (gradient moments loosely match observed stats)
    def generate_features(model_type, ic_type):
        if model_type == 'EW':
            # EW: lower variance, near-Gaussian
            base = np.random.randn(n_samples, 6) * 0.3
            base[:, 0] += 1.0  # m2 offset
        else:  # KPZ
            # KPZ: higher variance, skewed
            base = np.random.randn(n_samples, 6) * 0.5
            base[:, 0] += 1.5  # m2 offset
            base[:, 1] += 0.3  # m3 skewness
        
        # IC effect: droplet amplifies differences
        if ic_type == 'droplet':
            base *= 1.5  # Amplify all features
            base[:, 0] += 0.2  # Additional m2 shift
        
        return base
    
    # Generate 4 combinations
    ew_flat = generate_features('EW', 'flat')
    ew_droplet = generate_features('EW', 'droplet')
    kpz_flat = generate_features('KPZ', 'flat')
    kpz_droplet = generate_features('KPZ', 'droplet')
    
    # Concatenate
    features = np.vstack([ew_flat, ew_droplet, kpz_flat, kpz_droplet])
    
    # Labels
    class_labels = np.concatenate([
        np.zeros(n_samples),      # EW flat
        np.zeros(n_samples),      # EW droplet
        np.ones(n_samples),       # KPZ flat
        np.ones(n_samples)        # KPZ droplet
    ]).astype(int)
    
    domain_labels = np.concatenate([
        np.zeros(n_samples),      # flat
        np.ones(n_samples),       # droplet
        np.zeros(n_samples),      # flat
        np.ones(n_samples)        # droplet
    ]).astype(int)
    
    print(f"Generated synthetic data: {features.shape}")
    print(f"  EW: {(class_labels==0).sum()}, KPZ: {(class_labels==1).sum()}")
    print(f"  Flat IC: {(domain_labels==0).sum()}, Droplet IC: {(domain_labels==1).sum()}")
    
    return features, class_labels, domain_labels

# ============================================================================
# 5. TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, lambda_grl, epoch):
    model.train()
    total_loss = 0
    correct_class = 0
    correct_domain = 0
    total = 0
    
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (x, y_class, y_domain) in enumerate(dataloader):
        x = x.to(device)
        y_class = y_class.to(device)
        y_domain = y_domain.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        features, class_logits, domain_logits = model(x, lambda_grl)
        
        # Losses
        loss_class = class_criterion(class_logits, y_class)
        loss_domain = domain_criterion(domain_logits, y_domain)
        
        # Total loss (domain loss auto-reversed via GRL)
        loss = loss_class + loss_domain
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, pred_class = class_logits.max(1)
        _, pred_domain = domain_logits.max(1)
        correct_class += pred_class.eq(y_class).sum().item()
        correct_domain += pred_domain.eq(y_domain).sum().item()
        total += y_class.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'class_acc': 100. * correct_class / total,
        'domain_acc': 100. * correct_domain / total
    }

def validate(model, dataloader, lambda_grl):
    model.eval()
    correct_class = 0
    correct_domain = 0
    total = 0
    
    all_features = []
    all_class_labels = []
    all_domain_labels = []
    
    with torch.no_grad():
        for x, y_class, y_domain in dataloader:
            x = x.to(device)
            y_class = y_class.to(device)
            y_domain = y_domain.to(device)
            
            features, class_logits, domain_logits = model(x, lambda_grl)
            
            _, pred_class = class_logits.max(1)
            _, pred_domain = domain_logits.max(1)
            
            correct_class += pred_class.eq(y_class).sum().item()
            correct_domain += pred_domain.eq(y_domain).sum().item()
            total += y_class.size(0)
            
            all_features.append(features.cpu().numpy())
            all_class_labels.append(y_class.cpu().numpy())
            all_domain_labels.append(y_domain.cpu().numpy())
    
    all_features = np.vstack(all_features)
    all_class_labels = np.concatenate(all_class_labels)
    all_domain_labels = np.concatenate(all_domain_labels)
    
    return {
        'class_acc': 100. * correct_class / total,
        'domain_acc': 100. * correct_domain / total,
        'features': all_features,
        'class_labels': all_class_labels,
        'domain_labels': all_domain_labels
    }

# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 48: DOMAIN-ADVERSARIAL IC FACTORIZATION")
    print("="*70)
    
    # Setup
    output_dir = Path('results/exp48_domain_adversarial')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    features, class_labels, domain_labels = load_surface_growth_data()
    
    # Split train/val
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    train_dataset = SurfaceGrowthDataset(
        features[train_idx], class_labels[train_idx], domain_labels[train_idx]
    )
    val_dataset = SurfaceGrowthDataset(
        features[val_idx], class_labels[val_idx], domain_labels[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"\nDataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Model
    model = DomainAdversarialModel(input_dim=6, hidden_dim=16, feature_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training schedule: gradually increase λ (GRL strength)
    n_epochs = 100
    
    # Track history
    history = {
        'train_loss': [],
        'train_class_acc': [],
        'train_domain_acc': [],
        'val_class_acc': [],
        'val_domain_acc': []
    }
    
    print("\nTraining domain-adversarial network...")
    for epoch in range(n_epochs):
        # Anneal λ: start at 0, reach 1.0 at epoch 50
        lambda_grl = min(1.0, 2.0 * epoch / n_epochs)
        
        train_stats = train_epoch(model, train_loader, optimizer, lambda_grl, epoch)
        val_stats = validate(model, val_loader, lambda_grl)
        
        history['train_loss'].append(train_stats['loss'])
        history['train_class_acc'].append(train_stats['class_acc'])
        history['train_domain_acc'].append(train_stats['domain_acc'])
        history['val_class_acc'].append(val_stats['class_acc'])
        history['val_domain_acc'].append(val_stats['domain_acc'])
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | λ={lambda_grl:.2f} | "
                  f"Loss: {train_stats['loss']:.4f} | "
                  f"Class Acc: {val_stats['class_acc']:.1f}% | "
                  f"Domain Acc: {val_stats['domain_acc']:.1f}%")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    final_val = validate(model, val_loader, lambda_grl=1.0)
    
    print(f"\nValidation Accuracy:")
    print(f"  Class (EW vs KPZ):    {final_val['class_acc']:.2f}%")
    print(f"  Domain (flat vs drop): {final_val['domain_acc']:.2f}%")
    
    # Interpret results
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if final_val['class_acc'] > 90 and final_val['domain_acc'] < 60:
        print("✅ SUCCESS: Features encode universality class, blind to IC")
        print("   → IC-invariant coordinates discovered")
    elif final_val['class_acc'] > 90 and final_val['domain_acc'] > 90:
        print("❌ FAILURE: Features encode both class AND IC")
        print("   → Cannot factor out IC dependence with current observables")
    else:
        print("⚠️  INCONCLUSIVE: Poor class accuracy, model not converged")
    
    # PCA analysis
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(final_val['features'])
    
    # Correlation with class/domain
    r_class, p_class = pearsonr(features_2d[:, 0], final_val['class_labels'])
    r_domain, p_domain = pearsonr(features_2d[:, 0], final_val['domain_labels'])
    
    print(f"\nPC1 correlations:")
    print(f"  vs Class:  r = {r_class:.3f} (p = {p_class:.2e})")
    print(f"  vs Domain: r = {r_domain:.3f} (p = {p_domain:.2e})")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history['train_class_acc'], label='Train', alpha=0.7)
    ax.plot(history['val_class_acc'], label='Val', alpha=0.7)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Class Accuracy (EW vs KPZ)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(history['train_domain_acc'], label='Train', alpha=0.7)
    ax.plot(history['val_domain_acc'], label='Val', alpha=0.7)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance (goal)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Domain Accuracy (flat vs droplet)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Feature space by class
    ax = axes[1, 0]
    for class_idx, class_name in enumerate(['EW', 'KPZ']):
        mask = final_val['class_labels'] == class_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  label=class_name, alpha=0.5, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Feature Space by Universality Class')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Feature space by domain
    ax = axes[1, 1]
    for domain_idx, domain_name in enumerate(['flat', 'droplet']):
        mask = final_val['domain_labels'] == domain_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  label=domain_name, alpha=0.5, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Feature Space by Initial Condition')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confusion matrix style view
    ax = axes[1, 2]
    combinations = [
        (0, 0, 'EW + flat'),
        (0, 1, 'EW + droplet'),
        (1, 0, 'KPZ + flat'),
        (1, 1, 'KPZ + droplet')
    ]
    for class_idx, domain_idx, label in combinations:
        mask = (final_val['class_labels'] == class_idx) & (final_val['domain_labels'] == domain_idx)
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  label=label, alpha=0.6, s=30)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Feature Space by (Class, IC) Combination')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_dir / 'analysis.png'}")
    
    # Save results
    results = {
        'history': history,
        'final_val_class_acc': final_val['class_acc'],
        'final_val_domain_acc': final_val['domain_acc'],
        'features': final_val['features'],
        'class_labels': final_val['class_labels'],
        'domain_labels': final_val['domain_labels'],
        'pca_2d': features_2d,
        'r_class': r_class,
        'r_domain': r_domain,
        'p_class': p_class,
        'p_domain': p_domain
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
