# Research Plan: Universality Discovery via Anomaly Geometry

## 1. Scientific Motivation

### 1.1 The Limitation of Current Approaches

Standard universality class identification requires:
1. Propose a candidate theory (EW, KPZ, MBE, etc.)
2. Simulate or measure
3. Fit scaling exponents
4. Compare to theoretical predictions

This is **theory-first**: you must know what to look for.

### 1.2 The Inverse Problem

What if you have data from unknown dynamics? Current methods:
- Try all known theories, pick best fit
- Fail if dynamics are genuinely novel

**Our approach**: Let the data reveal structure through unsupervised learning.

### 1.3 Key Insight from Previous Work

The ml-universality project showed:
- Gradient features encode universality better than scaling exponents (12,591σ vs 0.43σ)
- But we *told* the model what features to use

**Question**: What features would the model discover on its own?

---

## 2. Technical Approach

### 2.1 Convolutional Autoencoder Architecture

**Input**: Raw height field h(x,t) as 2D image (space × time)
- Encodes full spatiotemporal structure
- No pre-computed features

**Encoder**: Convolutional layers → latent vector z
- Learns compressed representation
- Each dimension of z is a potential order parameter

**Decoder**: Transposed convolutions → reconstructed height field
- Forces z to retain essential information

**Training**: Reconstruction loss on EW/KPZ surfaces only
- Model learns "what normal looks like"
- Anomalies have high reconstruction error

### 2.2 Variational Autoencoder Extension

VAE adds:
- Probabilistic latent space
- KL divergence regularization
- Smoother interpolation between dynamics

Allows:
- Sampling new dynamics
- Measuring latent space distances as "universality distance"

### 2.3 Anomaly Score Definition

For test surface S:
```
A(S) = ||S - Decoder(Encoder(S))||² + β·D_KL(q(z|S) || p(z))
```

High A(S) = surface differs from training distribution

### 2.4 Clustering in Latent Space

1. Encode all test surfaces to latent vectors
2. Apply HDBSCAN (density-based clustering, handles noise)
3. Each cluster = candidate universality class
4. Characterize: What do cluster members share?

---

## 3. Experimental Design

### Experiment 1: Representation Discovery

**Goal**: Can autoencoder rediscover known universality structure?

**Method**:
1. Train on unlabeled EW + KPZ surfaces
2. Encode EW, KPZ, MBE, BD, VLDS surfaces
3. Visualize latent space with UMAP
4. Check if known classes form clusters

**Success criterion**: EW/KPZ overlap; MBE/BD/VLDS separated

### Experiment 2: Latent Dimension Interpretation

**Goal**: What do learned dimensions correspond to?

**Method**:
1. For each latent dimension z_i
2. Vary z_i while holding others fixed
3. Decode to surface
4. Measure physical properties (width, gradient, etc.)

**Expected**: Some dimensions correlate with known quantities; others may be novel

### Experiment 3: Anomaly Clustering

**Goal**: Do anomalies reveal unknown structure?

**Method**:
1. Generate 1000 surfaces from diverse dynamics (random parameters)
2. Encode all to latent space
3. Cluster with HDBSCAN
4. Analyze cluster membership vs. generation parameters

**Success criterion**: Clusters correspond to physically meaningful groupings

### Experiment 4: RG Basin Mapping

**Goal**: Map geometry of RG flow via anomaly scores

**Method**:
1. Define 2D parameter slice (e.g., λ vs κ in generalized KPZ)
2. Generate surfaces at each grid point
3. Compute anomaly score A(λ, κ)
4. Plot heatmap

**Expected**:
- Low A near EW/KPZ regions
- High A far from training distribution
- Sharp boundaries at phase transitions

### Experiment 5: Crossover Exponent Universality

**Goal**: Is γ (crossover sharpness) universal?

**Method**:
1. Extract D_ML(κ) for KPZ→MBE crossover
2. Extract D_ML(κ) for KPZ→EW crossover (vary ν instead)
3. Fit sigmoid/tanh to get γ values
4. Compare across crossover types

**Hypothesis**: Same γ for same universality classes, different for different

---

## 4. Potential Discoveries

### 4.1 Novel Universality Classes
If anomaly clusters don't match known classes, investigate:
- What dynamics generated them?
- Do they share scaling properties?
- Can we derive a governing equation?

### 4.2 Order Parameter Discovery
If latent dimension z_3 correlates with nothing known:
- It may be a new order parameter
- Physical interpretation needed
- Could guide theoretical development

### 4.3 RG Geometry
If basin boundaries are measurable:
- Compare to perturbative RG predictions
- Identify regions where perturbation theory fails
- Map non-perturbative structure

### 4.4 Crossover Classification
If crossovers have characteristic exponents:
- New classification scheme for phase transitions
- Universality beyond fixed points

---

## 5. Technical Requirements

### Compute
- GPU for autoencoder training (RTX 3080 or better)
- ~10k training surfaces, ~10k test surfaces
- Training time: ~2-4 hours per model

### Software
- PyTorch for neural networks
- HDBSCAN for clustering
- UMAP for visualization
- Optuna for hyperparameter optimization

### Data
- Reuse simulation code from ml-universality-classification
- Extend parameter ranges
- Add new dynamics (random/exotic PDEs)

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Autoencoder doesn't separate known classes | Try different architectures, more data |
| Latent space uninterpretable | Use disentangled VAE (β-VAE) |
| No meaningful clusters | May indicate continuous spectrum, not failure |
| Compute limitations | Start with 1D surfaces, scale up |

---

## 7. Publication Strategy

**Target journals** (in order of ambition):
1. Nature Physics - if RG basin mapping works and reveals new physics
2. Physical Review Letters - if crossover universality is confirmed
3. Physical Review E - solid methodology paper
4. Machine Learning: Science and Technology - ML methods focus

**Timeline**:
- Month 1-3: Core experiments
- Month 4: Analysis and interpretation
- Month 5: Writing
- Month 6: Submission
