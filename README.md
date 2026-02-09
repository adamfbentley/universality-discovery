# Universality Discovery via Anomaly Geometry

**Unsupervised discovery of universality classes through representation learning and anomaly clustering**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Vision

Move from **testing known theories** to **discovering unknown structure**.

| Approach | Question |
|----------|----------|
| Traditional physics | "Does my data fit KPZ theory?" |
| Previous work | "Is this system non-KPZ?" |
| **This project** | "What theories exist? Let the data reveal them." |

## Core Ideas

### 1. Unsupervised Representation Learning
Instead of hand-crafted features (gradients, width, exponents), train autoencoders to discover **what matters** for distinguishing dynamics. The learned latent space becomes a theory-agnostic coordinate system.

### 2. Anomaly Geometry Maps RG Flow
- **Low anomaly regions** = basins of attraction around RG fixed points
- **Sharp boundaries** = separatrices between universality classes
- **Anomaly clusters** = potentially undiscovered universality classes

### 3. Crossover Universality
The *shape* of transitions between universality classes (D_ML(κ) curves) may itself be universal. Different crossovers may have characteristic exponents.

## Research Questions

1. Can autoencoders rediscover known universality class structure without supervision?
2. Do anomalies cluster into coherent groups representing unknown dynamics?
3. Can we map RG basin boundaries by scanning parameter space?
4. Is there a "universality of crossovers" with its own exponents?
5. What latent dimensions correspond to physical order parameters?

## Project Structure

```
universality-discovery/
├── src/
│   ├── models/           # Autoencoders, VAEs, anomaly detectors
│   ├── simulation/       # Surface growth models (reuse from ml-universality)
│   ├── exploration/      # Parameter space scanning
│   └── analysis/         # Clustering, RG boundary detection
├── experiments/          # Numbered experiment scripts
├── notebooks/            # Exploratory analysis
├── figures/              # Generated visualizations
└── docs/                 # Research notes and findings
```

## Roadmap

### Phase 1: Representation Learning (Weeks 1-3)
- [ ] Implement convolutional autoencoder for surface height fields
- [ ] Train on EW/KPZ surfaces without labels
- [ ] Verify latent space separates known classes
- [ ] Analyze what latent dimensions encode

### Phase 2: Anomaly Clustering (Weeks 4-6)
- [ ] Generate diverse test dynamics (MBE, VLDS, BD, quenched-KPZ, etc.)
- [ ] Map all to learned latent space
- [ ] Cluster anomalies using HDBSCAN
- [ ] Characterize clusters - what physics do they share?

### Phase 3: RG Basin Mapping (Weeks 7-9)
- [ ] Define continuous parameter space (ν, λ, κ, noise amplitude, etc.)
- [ ] Compute anomaly score across parameter grid
- [ ] Identify basin boundaries and critical surfaces
- [ ] Compare to known RG predictions

### Phase 4: Crossover Universality (Weeks 10-12)
- [ ] Extract D_ML(parameter) curves for multiple crossovers
- [ ] Fit crossover exponents (κ_c, γ)
- [ ] Test if exponents are universal within crossover types
- [ ] Develop theoretical framework for crossover classification

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
torch  # For autoencoders
hdbscan  # For anomaly clustering
umap-learn  # For visualization
```

## Relationship to Previous Work

This extends [ml-universality-classification](https://github.com/adamfbentley/ml-universality-classification) from a **diagnostic tool** to a **discovery engine**.

| ml-universality-classification | universality-discovery |
|-------------------------------|------------------------|
| Hand-crafted features | Learned representations |
| Binary anomaly detection | Anomaly clustering |
| Single crossover study | RG basin mapping |
| Method validation | Theory discovery |

## License

MIT
