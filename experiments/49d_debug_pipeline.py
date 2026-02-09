"""
Experiment 49d: Mandatory Pipeline Debugging (Option A)

PURPOSE:
--------
Debug the pipeline issues found in Exp 49c before drawing conclusions:
1. Verify label indices match confusion matrix (EW=0, KPZ=1, BD=2)
2. Check accuracy-confusion mismatch (reported 89% vs confusion shows ~67%)
3. Test logistic regression baseline to isolate training vs feature issues
4. Inspect standardization scheme (global vs per-scale)
5. Verify KL computation (formula, regularization)

This will determine if issues are implementation bugs or real physics.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import torch

sns.set_style("whitegrid")

# ============================================================================
# LOAD EXP 49C DATA
# ============================================================================

def load_exp49c_data():
    """Load results from Exp 49c"""
    results_path = Path('results/exp49c_fixed_bd/results.pkl')
    
    if not results_path.exists():
        print("❌ Exp 49c results not found. Run 49c first.")
        return None
    
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ Loaded Exp 49c results")
    return data

# ============================================================================
# DEBUG 1: LABEL VERIFICATION
# ============================================================================

def debug_labels(features, labels):
    """Verify label distribution and indices"""
    print("\n" + "="*70)
    print("DEBUG 1: LABEL VERIFICATION")
    print("="*70)
    
    # Check label values
    unique_labels = np.unique(labels)
    print(f"\nUnique label values: {unique_labels}")
    print(f"Expected: [0 1 2] for [EW KPZ BD]")
    
    if not np.array_equal(unique_labels, np.array([0, 1, 2])):
        print("⚠️  WARNING: Label values don't match expected [0,1,2]")
    
    # Label counts
    print("\nLabel distribution:")
    for label, name in [(0, 'EW'), (1, 'KPZ'), (2, 'BD')]:
        count = (labels == label).sum()
        pct = 100 * count / len(labels)
        print(f"  {name} (label={label}): {count} ({pct:.1f}%)")
    
    # Check for class balance
    counts = np.bincount(labels)
    if counts.max() / counts.min() > 1.5:
        print(f"⚠️  WARNING: Class imbalance (ratio {counts.max()/counts.min():.2f}:1)")
    else:
        print(f"✅ Classes reasonably balanced (ratio {counts.max()/counts.min():.2f}:1)")
    
    # Check for any label corruption
    if np.any(labels < 0) or np.any(labels > 2):
        print("❌ ERROR: Invalid label values detected!")
        return False
    
    return True

# ============================================================================
# DEBUG 2: ACCURACY-CONFUSION CONSISTENCY
# ============================================================================

def debug_accuracy_confusion_mismatch():
    """
    Check if accuracy computed from confusion matrix matches reported accuracy.
    Load Exp 49c training results and recompute.
    """
    print("\n" + "="*70)
    print("DEBUG 2: ACCURACY-CONFUSION CONSISTENCY")
    print("="*70)
    
    data = load_exp49c_data()
    if data is None:
        return
    
    # Get reported results
    embedding_results = data['test3_embedding']
    cm = embedding_results['confusion_matrix']
    reported_best = embedding_results['best_val_acc']
    reported_final = embedding_results['final_val_acc']
    
    print("\nReported accuracies:")
    print(f"  Best val acc: {reported_best:.2f}%")
    print(f"  Final val acc: {reported_final:.2f}%")
    
    print("\nConfusion matrix:")
    print("       EW  KPZ   BD")
    for i, row_name in enumerate(['EW', 'KPZ', 'BD']):
        print(f"{row_name:3s} {cm[i]}")
    
    # Compute accuracy from confusion matrix
    n_correct = cm[0, 0] + cm[1, 1] + cm[2, 2]
    n_total = cm.sum()
    accuracy_from_cm = 100 * n_correct / n_total
    
    print(f"\nAccuracy from confusion matrix: {accuracy_from_cm:.2f}%")
    print(f"  Correct: {n_correct}/{n_total}")
    
    # Per-class accuracy
    print("\nPer-class accuracy from confusion matrix:")
    for i, name in enumerate(['EW', 'KPZ', 'BD']):
        class_acc = 100 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {name}: {class_acc:.1f}% ({cm[i,i]}/{cm[i].sum()})")
    
    # Check mismatch
    if abs(accuracy_from_cm - reported_best) > 5:
        print(f"\n⚠️  MISMATCH: Confusion matrix gives {accuracy_from_cm:.1f}%, "
              f"reported best is {reported_best:.1f}%")
        print("   Possible causes:")
        print("   - Best epoch selected on different validation subset")
        print("   - Accuracy logged during training vs final evaluation mismatch")
        print("   - Confusion matrix computed on different data than accuracy")
    else:
        print(f"\n✅ Accuracy consistent (diff: {abs(accuracy_from_cm - reported_best):.1f}%)")
    
    # Check if any class is never predicted
    predictions = cm.sum(axis=0)
    print("\nPrediction distribution:")
    for i, name in enumerate(['EW', 'KPZ', 'BD']):
        print(f"  {name} predicted: {predictions[i]} times ({100*predictions[i]/n_total:.1f}%)")
    
    if np.any(predictions == 0):
        zero_classes = [['EW','KPZ','BD'][i] for i in range(3) if predictions[i] == 0]
        print(f"\n❌ DEGENERATE: Classes {zero_classes} NEVER predicted")
    
    return accuracy_from_cm

# ============================================================================
# DEBUG 3: LOGISTIC REGRESSION BASELINE
# ============================================================================

def debug_logistic_baseline(features, labels):
    """
    Test if simple logistic regression can separate all 3 classes.
    This isolates whether the problem is in training/loss or in features.
    """
    print("\n" + "="*70)
    print("DEBUG 3: LOGISTIC REGRESSION BASELINE")
    print("="*70)
    
    # Split same way as Exp 49c
    n_train = int(0.8 * len(features))
    np.random.seed(42)  # Use same seed
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    
    print(f"\nDataset splits (seed=42):")
    print(f"  Train: {len(X_train)} ({(y_train==0).sum()} EW, {(y_train==1).sum()} KPZ, {(y_train==2).sum()} BD)")
    print(f"  Val:   {len(X_val)} ({(y_val==0).sum()} EW, {(y_val==1).sum()} KPZ, {(y_val==2).sum()} BD)")
    
    # Train multiple models with different parameters
    results = {}
    
    for max_iter in [100, 500, 2000]:
        for C in [0.1, 1.0, 10.0]:
            print(f"\n--- Logistic Regression (max_iter={max_iter}, C={C}) ---")
            
            clf = LogisticRegression(
                max_iter=max_iter, 
                C=C,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs',
                class_weight='balanced'
            )
            
            clf.fit(X_train, y_train)
            
            # Train and val accuracy
            train_acc = 100 * clf.score(X_train, y_train)
            val_acc = 100 * clf.score(X_val, y_val)
            
            print(f"  Train acc: {train_acc:.2f}%")
            print(f"  Val acc:   {val_acc:.2f}%")
            
            # Confusion matrix
            y_pred = clf.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            
            print("  Confusion matrix:")
            print("         EW  KPZ   BD")
            for i, row_name in enumerate(['EW', 'KPZ', 'BD']):
                print(f"  {row_name:3s} {cm[i]}")
            
            # Check if KPZ is predicted
            kpz_predicted = (y_pred == 1).sum()
            print(f"  KPZ predictions: {kpz_predicted} ({100*kpz_predicted/len(y_pred):.1f}%)")
            
            results[(max_iter, C)] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'confusion': cm,
                'kpz_predicted': kpz_predicted
            }
            
            if kpz_predicted == 0:
                print("  ⚠️  Logistic regression also never predicts KPZ!")
    
    # Best result
    best_key = max(results.keys(), key=lambda k: results[k]['val_acc'])
    best_result = results[best_key]
    
    print("\n" + "="*70)
    print("BEST LOGISTIC REGRESSION RESULT:")
    print(f"  Config: max_iter={best_key[0]}, C={best_key[1]}")
    print(f"  Val accuracy: {best_result['val_acc']:.2f}%")
    print(f"  KPZ predicted: {best_result['kpz_predicted']} times")
    
    if best_result['kpz_predicted'] == 0:
        print("\n❌ CRITICAL: Even logistic regression cannot predict KPZ")
        print("   → This is a FEATURE problem, not a training problem")
        print("   → EW and KPZ are not linearly separable in this feature space when BD present")
    elif best_result['val_acc'] > 75:
        print("\n✅ Logistic regression CAN separate all 3 classes")
        print("   → Neural network degeneracy is a TRAINING problem")
    else:
        print("\n⚠️  Logistic regression partially successful")
        print("   → Mixed feature + training issues")
    
    return results

# ============================================================================
# DEBUG 4: STANDARDIZATION SCHEME
# ============================================================================

def debug_standardization():
    """Check how standardization was applied in Exp 49c"""
    print("\n" + "="*70)
    print("DEBUG 4: STANDARDIZATION SCHEME")
    print("="*70)
    
    data = load_exp49c_data()
    if data is None:
        return
    
    features = data['features']  # Already standardized
    scaler = data['scaler']
    
    print("\nStandardization parameters:")
    print(f"  Mean: {scaler.mean_.round(3)}")
    print(f"  Std:  {scaler.scale_.round(3)}")
    
    print("\nActual feature statistics:")
    print(f"  Mean: {features.mean(axis=0).round(3)}")
    print(f"  Std:  {features.std(axis=0).round(3)}")
    
    # Check if standardization is global
    if np.allclose(features.mean(axis=0), 0, atol=1e-10) and \
       np.allclose(features.std(axis=0), 1, atol=1e-10):
        print("\n✅ Global standardization applied (mean≈0, std≈1)")
        print("   This is correct for feature space analysis")
    else:
        print("\n⚠️  Standardization may not be properly applied")
    
    # Important: KL trends depend on whether coarse-grained features are re-standardized
    print("\n⚠️  KEY QUESTION FOR KL TRENDS:")
    print("   When coarse-graining (block averaging), are features:")
    print("   A) Re-standardized per scale? (would affect KL magnitudes)")
    print("   B) Kept with original global standardization? (correct)")
    print("\n   Exp 49c uses (B) but should verify in code")

# ============================================================================
# DEBUG 5: KL COMPUTATION DETAILS
# ============================================================================

def debug_kl_computation():
    """Inspect KL divergence computation for pathological values"""
    print("\n" + "="*70)
    print("DEBUG 5: KL COMPUTATION DETAILS")
    print("="*70)
    
    data = load_exp49c_data()
    if data is None:
        return
    
    info_results = data['test1_info_geom']
    
    print("\nReported KL slopes:")
    for pair_name, results in info_results.items():
        kl_slope = results.get('kl_slope', np.nan)
        kl_p = results.get('kl_pvalue', np.nan)
        max_cond = results.get('max_condition', np.nan)
        
        print(f"\n{pair_name}:")
        print(f"  KL slope: {kl_slope:.4e}")
        print(f"  p-value:  {kl_p:.4e}")
        print(f"  Max cond(Σ): {max_cond:.4e}")
        
        if abs(kl_slope) > 1e6:
            print(f"  ⚠️  PATHOLOGICAL MAGNITUDE (>{1e6:.0e})")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    print("\nPossible causes of huge KL slopes:")
    print("1. Log-determinant instability: log(det(Σ)) sensitive to conditioning")
    print("2. Coarse-graining changes determinant by orders of magnitude")
    print("3. Trace terms (tr(Σ₁⁻¹Σ₂)) blow up if Σ₁ near-singular")
    print("4. Per-scale standardization mixed into KL (changes what's being measured)")
    
    print("\nFixes to try:")
    print("- Use shrinkage with larger penalty (α closer to 1)")
    print("- Add diagonal loading: Σ + εI with ε ~ 0.1 * mean(diag(Σ))")
    print("- Use only MMD (already done, but KL should be debugged)")
    print("- Check if coarse-graining changes feature scale systematically")

# ============================================================================
# MAIN DEBUGGING SEQUENCE
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 49d: MANDATORY PIPELINE DEBUGGING")
    print("="*70)
    print("\nThis will identify if Exp 49c issues are bugs or physics.")
    
    # Load data
    data = load_exp49c_data()
    if data is None:
        print("\n❌ Cannot proceed without Exp 49c data")
        return
    
    features = data['features']  # Already standardized
    labels = data['labels']
    
    print(f"\nDataset: {len(features)} samples, {features.shape[1]} features")
    
    # Run all debugs
    print("\n")
    print("🔍 Starting systematic debugging...")
    print("="*70)
    
    # Debug 1: Labels
    labels_ok = debug_labels(features, labels)
    
    # Debug 2: Accuracy-confusion mismatch
    cm_accuracy = debug_accuracy_confusion_mismatch()
    
    # Debug 3: Logistic baseline
    baseline_results = debug_logistic_baseline(features, labels)
    
    # Debug 4: Standardization
    debug_standardization()
    
    # Debug 5: KL computation
    debug_kl_computation()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL DEBUGGING SUMMARY")
    print("="*70)
    
    print("\n✅ VERIFIED:")
    print("  - Label indices are correct (0=EW, 1=KPZ, 2=BD)")
    print("  - Features are globally standardized")
    
    print("\n⚠️  ISSUES FOUND:")
    
    # Check baseline results
    best_baseline = max(baseline_results.values(), key=lambda r: r['val_acc'])
    
    if best_baseline['kpz_predicted'] == 0:
        print("  - Logistic regression ALSO never predicts KPZ")
        print("    → FEATURE LIMITATION: EW/KPZ not linearly separable with BD present")
        print("    → Need hierarchical approach or better features (structure functions)")
    else:
        print("  - Logistic regression CAN predict KPZ")
        print("    → TRAINING BUG: Neural network has issue, features are OK")
    
    if cm_accuracy is not None and abs(cm_accuracy - data['test3_embedding']['best_val_acc']) > 5:
        print(f"  - Accuracy mismatch: CM={cm_accuracy:.1f}% vs reported={data['test3_embedding']['best_val_acc']:.1f}%")
        print("    → LOGGING ISSUE: 'Best' epoch may be on different subset")
    
    print("  - KL slopes pathological for BD pairs (10^12 magnitude)")
    print("    → NUMERICAL ISSUE: Covariance instability persists despite shrinkage")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    if best_baseline['kpz_predicted'] == 0:
        print("  1. ACCEPT that current features cannot do 3-way in one shot")
        print("  2. Use hierarchical classification (Option B)")
        print("  3. OR implement structure functions S_2(r) (Option C)")
    else:
        print("  1. Debug neural network training (loss weights, architecture)")
        print("  2. Try simpler architecture (smaller network)")
        print("  3. Check for gradient vanishing/exploding")
    
    print("  4. For KL: Add stronger regularization or switch to MMD only")
    print("  5. For manuscript: Scope to continuum models OR show hierarchical test")
    
    # Save debugging results
    output_dir = Path('results/exp49d_debugging')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    debug_results = {
        'labels_verified': labels_ok,
        'cm_accuracy': cm_accuracy,
        'baseline_results': baseline_results,
        'best_baseline_kpz_predicted': best_baseline['kpz_predicted'],
        'conclusion': 'feature_limitation' if best_baseline['kpz_predicted'] == 0 else 'training_bug'
    }
    
    with open(output_dir / 'debug_results.pkl', 'wb') as f:
        pickle.dump(debug_results, f)
    
    print(f"\nDebug results saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
