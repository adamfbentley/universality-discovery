# Experiment Log

Working notes from the universality discovery project. Roughly chronological — older interpretations are left as-is even when later work revised them, with corrections noted up top.

### Corrections added later

- Feb 11: "PC1 = universality axis" from Exp 21 is overstated. Exp 27 showed it's IC-dependent, and when varying KPZ params, PC1 is dominated by variance features scaling with D/nu. Treat it as a good protocol-specific discriminator, not a universal coordinate.
- Feb 11: Potts Exp 55 was invalid — used the Ising Wolff bond probability instead of the 3-state Potts form. Everything from 55b onward is fine.
- Feb 9: KPZ coupling coordinate corrected. PC1 tracks D/nu (not D/nu^3 as first reported). Follows from the known stationary slope measure (Exp 54). So it's an amplitude coordinate, potentially blind to lambda at stationarity.
- Feb 6: Tracy-Widom claim from Exp 26 downgraded. Single-point skewness isn't specific — EW also showed TW-like values. Consistency check only, not a proof.
- The "RG merges manifolds" result from Exp 23 depends heavily on normalization choices. Not as clean as originally presented.

---

## Autoencoder exploration (Jan 2026, Exp 1-12)

### Exp 1: Representation learning

Trained autoencoder on 200 EW + 200 KPZ surfaces, then tested on BD, Eden, RD. Discrete models are massively anomalous — BD at 1400x, Eden 260x, RD 2500x. The problem is Eden is supposed to be in the KPZ universality class, so it really shouldn't look 260x different. AE is picking up on discreteness of the growth rules, not the actual universality structure.

### Exp 2: Latent dimension interpretation

Tried to figure out what the AE latent dims correspond to. Perturbed each one, checked correlations with width, roughness exponent, gradient stats. Nothing maps cleanly to known physics quantities. It's optimizing compression, not learning anything interpretable.

### Exp 3: Latent space clustering

Ran UMAP + HDBSCAN on the latent codes. EW and KPZ overlap (both in training set so that's expected). Discrete models each form tight separate clusters. The whole space organizes by how the surfaces were generated, not by universality class. Not what we wanted.

### Exp 4: Long-time Eden

Thought maybe if we ran Eden longer, it'd converge toward KPZ in the AE's view. Tested T=500, 1000, 1500. The Eden/KPZ separation ratio stays dead flat around 730-764x. Discreteness signature does not wash out with time. Surprising and a bit discouraging.

### Exp 5: Gradient space + coarse-graining

Following a suggestion to work in gradient space and blur. The Eden/BD ratio does drop nicely from 14.5x to 2x at sigma=2, so within KPZ class things converge somewhat. But absolute anomaly scores stay enormous — 1000x+ for discrete models even at optimal blur. Also non-monotonic at high sigma which is weird.

### Exp 6: Train on discrete (reverse paradigm)

Flipped the whole approach — trained AE on BD+EDEN instead. Continuum models come out at 0.01x anomaly, model handles them trivially. RD correctly flagged at 2.9x. This was the first real "aha" moment. A model trained on the harder (discrete) examples generalises to the easy ones, but not the other way round. Makes intuitive sense.

### Exp 7 / 7b: EW vs KPZ discrimination + Wasserstein geometry

The discrete-trained model distinguishes EW from KPZ with p < 10^-158. Cohen's d = 5.13. Three-level hierarchy: implementation type (continuum/discrete), then universality class, then specific model. Wasserstein distances in latent space confirm the structure quantitatively — EW-KPZ very close (d=1.18), BD-EDEN compact (d=6.65), RD distant from everything.

### Exp 8: Physics-informed autoencoder

Added a beta predictor head to the AE loss. Latent clustering improves 167x but the actual EW/KPZ class discrimination ratio barely moves (1.02x). The bottleneck isn't latent structure — gradient features genuinely don't distinguish the two classes well enough through the AE.

### Exp 9-11: Wavelets and gradient features

Tried wavelet decomposition on W(t) (Exp 9), skewness/kurtosis features (Exp 10), and wavelet decomposition on spatial gradients (Exp 11). None of them manage to separate EW from KPZ while grouping KPZ-class models together. EW-KPZ distance stays tiny in most representations. The continuum/discrete gap dominates everything. In Exp 10, all models show skewness near 0, nowhere close to the TW target of 0.29 — simulations haven't reached asymptotic regime.

### Exp 12: RG-geodesic metric learning

Implemented Cotler-Rezchikov style geodesic loss. All pairwise distances collapsed to about 0.03-0.05 — the smoothness regularisation wiped out any class structure. Over-constrained: trying to learn both the metric and the RG flow at once without knowing the fixed points.

---

## Diagnostics and information geometry (Jan 2026, Exp 13-19)

### Exp 13: Slope-growth coupling

The experiment that changed the direction of the project. Directly regressed local growth against slope-squared to detect the KPZ nonlinearity lambda*(grad h)^2. Continuum KPZ gives b=+0.027, clearly positive (p=0.027). EW gives b\~0, RD gives b\~0. Discrete models show weaker signal. The nonlinearity IS detectable — we just weren't looking for it correctly with the autoencoders. This explains a lot of the earlier failures.

### Exp 14: Coarse-grained slope-growth

Tried to reveal the nonlinearity in discrete models via spatial coarse-graining. All models show b DECREASING with blur — the opposite of what we wanted. Gaussian smoothing destroys the signal instead of enhancing it. Simple spatial blur isn't a proper RG transformation (no time rescaling, no coupling renormalization). Disappointing after Exp 13's success.

### Exp 15: Information geometry

Computed Fisher information matrices and Ricci scalar curvature for observable triplets. Found something unexpected: discrete models have HIGH curvature that flows down under coarse-graining, continuum models sit near R=0 at all scales. BD goes from R=6.18 to 0.18 (35x reduction), RD from 351 to 1.4. Discreteness literally IS information-geometric curvature. The curvature flow under CG is monotonic. Pretty interesting connection.

### Exp 16: Validation of info geometry

Tested robustness after concerns about KDE sensitivity. Absolute R values are unstable (CV > 0.6) across bandwidth choices, but the ordering BD >> EDEN >> EW \~ KPZ holds at every bandwidth tested. The qualitative finding from Exp 15 survives, just can't trust the absolute numbers.

### Exp 17-19: Total correlation saga

Tried TC as a cleaner information-theoretic metric (Exp 17). Initial results looked great — basin structure in TC space. Then it fell apart. Exp 18: the k-NN estimator was producing negative TC values (mathematically impossible). After fixing with clipping and null tests, using a common dynamical exponent z for all models makes the EW-KPZ separation vanish. The basin structure was an artifact of injecting class knowledge through class-specific exponents. Exp 19: even with proper i.i.d. sampling, TC of local pointwise observables is essentially zero for all models. The signal genuinely isn't there — local observables are nearly independent by construction. Dead end, but at least definitively so.

---

## Manifold structure (Jan 2026, Exp 20-27)

### Exp 20: Intrinsic dimension

Measured intrinsic dimension of feature manifolds using PCA, MLE, and TwoNN estimators. All three agree: d\~2 for both EW and KPZ in the 6D gradient moment space. BD is higher around d\~5. The manifolds really are low-dimensional.

### Exp 21: Coordinates of universality

PCA on the 6D gradient features. PC1 loads on grad_var, lap_var, h_var and correlates with model identity at r=-0.956 — near perfect separation. PC2 loads on skewness/kurtosis. T and L show essentially no correlation with either axis (r\~0.05), so the universality info is orthogonal to finite-size effects. Seemed like a huge result at the time, but Exp 27 later showed this is IC-dependent.

### Exp 22: Robustness tests

BD/Eden don't project onto the KPZ cluster (10000x higher gradient variance), so discrete models live in a completely different region. But separation is stable across L and T (CV=0.08), and logistic regression gets AUC=1.000 on raw features. The coordinates work within the continuum family.

### Exp 23: Discrete-continuum gap

Diagnosed the gap: scale-dependent variance features are the culprit. Applied block coarse-graining and KPZ-BD distance drops 90% (2.34 to 0.26). RG working as expected — microscopics wash out, universal structure emerges. Though the normalization matters more than I initially appreciated (see errata).

### Exp 24: Differential contraction

Was supposed to be the "killer plot" — BD->KPZ contracts while EW-KPZ stays constant under RG. Instead: BD->KPZ contracts only \~15%, and EW-KPZ EXPANDS by 45%. Not the clean result at all. Turns out gradient moments are RG-relevant operators — they grow under coarse-graining instead of staying constant. Scientifically interesting (tells you about operator structure) but not the validation I was looking for.

### Exp 26: Tracy-Widom validation

KPZ single-point height fluctuation skewness comes out at -0.297, matching the TW theoretical value of -0.29 to 2%. Nice sanity check that the simulations are correct. But EW also shows skewness -0.299, which is unexpected — finite-size boundary effects. Height statistics alone don't discriminate; that's why gradient features are necessary. Later downgraded from "breakthrough" to "consistency check" (see corrections).

### Exp 27: Initial condition control test

This was big. Tested PC1 separation across flat, droplet, and stationary ICs. Droplet gives r=-0.982 (perfect separation), flat gives r=-0.06 (nothing), stationary is marginal. The PC1 loading vector itself rotates between IC types (cosine similarity 0.12 between flat and droplet — nearly orthogonal). Universality class detection is IC-dependent, connecting to the GOE/GUE/Baik-Rains structure in KPZ fixed point theory. More complex than the simple "universal axis" story from Exp 21, but also more interesting. The separation comes from non-equilibrium dynamics, not equilibrium statistics.

---

## Time-resolved studies and RG operator tests (Exp 28-36)

### Exp 28-30: RG operator variants

Time-resolved flat IC confirms PC1 never gets strong (peak r\~0.325, Exp 28). Tested various RG operators — baseline spatial gives BD->KPZ \~14% contraction, EW-KPZ \~26% drift (Exp 29). Adding time/height rescaling causes blow-ups. Gaussian smoothing gives +261% EW-KPZ drift (Exp 30). None of these are faithful RG surrogates.

### Exp 31-36: Tracy-Widom deep dive

Several attempts at proper TW validation. Global skewness doesn't converge stably (31). Dimensionless features still drift under CG (32). The "proper TW" measurement stays noisy at accessible L, T (33). Identified scale-separation requirement L >> sqrt(T) (34). Better results with large L (35, KPZ skew +0.223) but still not rigorous. Droplet TW-GUE attempt started but not finished (36).

---

## Embedding methods and coupling coordinates (Exp 45-48)

### Exp 45/45b: RG-covariant embedding

Self-supervised RG-covariance loss collapses to trivial solution (45). Anti-collapse constraints fix it (45b), but you have to be careful about what the task regularisation is actually doing vs what the RG loss contributes.

### Exp 46/46b: KPZ coupling coordinate

PC1 tracks D/nu with r\~0.961. Originally reported as D/nu^3 but later corrected via Exp 54 — it's the simpler quantity, explainable from the known 1D stationary slope measure. Strong result but it's an amplitude coordinate, potentially lambda-blind at stationarity.

### Exp 47: Info-geometric distances vs scale

KL and Bhattacharyya distances between EW and KPZ increase with coarse-graining. This is the RG-relevance signature — gradient features really are relevant operators. Cleaner evidence than the TC approach from Exp 17-19.

### Exp 48: Domain-adversarial IC factorisation

Tried adversarial training to factor out IC while keeping class info. Partially worked (class accuracy \~78%, domain accuracy \~65%). Didn't fully achieve the separation.

---

## BD debug and KS campaign (Exp 49-50 series)

### Exp 49 series: BD generalisation

Four experiments debugging why the neural net fails on BD. Punchline from 49d: logistic regression gets 100% accuracy on the same features. The representation is fine — it was a training bug in the DL pipeline.

### Exp 50 series: KS generalisation (long campaign)

The biggest campaign in the project, spanning \~20 sub-experiments. Started with "does KS converge to KPZ?" (no), then went through multiple rounds where exciting slopes turned out to be pipeline artifacts — bandwidth drift, feature collapse, ordering problems.

The diagnostic gate concept was forged here: check same-class consistency before trusting any cross-class result. Exp 50k/50l show KS stays separate from KPZ under valid diagnostics (flat distance). Exp 50n supports KS != KPZ with scale-free structure functions. The critical conceptual result came from Exp 50q/50r: scale-invariant observables can still encode non-universal amplitudes (50q), but restricting to exponent-only observables gives correct same-class agreement (50r). This distinction between "scale-invariant" and "universal" was important.

---

## Cross-system validation (Exp 51-54)

### Exp 51: RG-covariant autoencoder

Semigroup property passes but transfer doesn't cleanly discriminate classes. Needs contrastive constraints to work as a universality detector.

### Exp 52 series: Ising

PC1 tracks reduced temperature at r\~0.971, even after dropping |m| and E/N (52b). FSS collapse gives nu=1.073, about 7% error (52d). This is the gold standard result — unsupervised features recovering quantitative critical exponents. Two-parameter scan (52e) clearly shows the thermal direction but the magnetic axis is invisible with Z2-even features, which makes sense.

### Exp 53/53b: Vicsek

PC1 tracks noise eta at r\~0.958 (0.926 without local phi features). Strong empirically but lacks an FSS-level validation like we got for Ising.

### Exp 54: Theoretical validation

Confirmed numerically that Var[dh/dx] \~ D/nu_KPZ and is lambda-independent at stationarity. Explains why PC1 works and its limitations.

---

## Potts campaign (Exp 55-60)

### Exp 55/55b: Potts FSS

First attempt invalid (wrong Wolff bond prob). Corrected version (55b) still can't stably extract nu from PC1 via the window-variance collapse metric. This is where the PCA-FSS method hits its boundary.

### Exp 56 series: Potts diagnostics

Four experiments probing the failure: symmetric features, scaling dimension tests, slope scaling, width scaling. The issue seems to be metric degeneracy plus Z3 symmetry complications combined with corrections to scaling. No single fix resolved it.

### Exp 57 series: Binder controls

Standard Binder cumulant analysis works for Potts — 57c gets nu\~0.884 (\~6% error from the exact 5/6). This suggests the physics is accessible, while the failure is specific to PCA-FSS rather than Potts transitions in general.

### Exp 58-59: Symmetry sector attempts

Adding order parameter components didn't rescue the Potts nu identification. Exp 59 gives nu_opt \~ 2.51, clearly wrong. The method needs more fundamental observable/metric surgery for generality beyond Ising.

### Exp 60: Fisher/sensitivity analysis

Gaussian-approximation sensitivity matrix a la Machta's sloppy models. KPZ result is solid: dominant eigenvalue aligns with D/nu at cosine \~0.999. Ising equally strong: mean-tangent aligns with Exp 52b PC1 at cos\~0.999. The Potts comparison was invalid — compared different mathematical objects across systems. Core KPZ and Ising results hold, but the cross-system diagnostic wasn't properly established.

---

## Return to discovery vision (Exp 61-62)

### Exp 61: Autoencoder anomaly geometry

Full circle — tried the original project vision of AE on known classes, cluster unknowns. With L=64, T=200, 30 training samples: HDBSCAN finds 0 clusters. KMeans ARI=0.002. kNN accuracy 58%. KS has the lowest reconstruction error because its surfaces are actually smoother (4th-order dissipation). Massive overfitting: 1.8M params for 60 training samples. The AE approach just doesn't work here.

### Exp 62: Feature-space clustering

Skipped the autoencoder entirely, clustered directly on the 6D gradient features used throughout the earlier diagnostics. This is the honest synthesis — use the feature family that had survived the most checks.

Pilot (L=128, T=500, 30 samples/system, 180 total): HDBSCAN finds exactly 4 clusters matching the 4 true universality classes. ARI=0.496, kNN 83.9%. Dramatic improvement over Exp 61.

Full run (L=256, T=2000, 80 samples/system, 480 total): HDBSCAN still finds 4 clusters, ARI=0.495, 3-NN=82.1%. Essentially identical to pilot — more data didn't help. In these runs, ARI\~0.495 looks like a persistent ceiling for this feature family rather than a sample size problem. The EW-KPZ centroid distance actually decreased from 1.04 to 0.87 at longer simulation times, which makes sense — diffusion dominates at late times, making the two classes look more similar. 4/5 quantitative predictions passed; the one failure (ARI>0.5) is frustratingly close.

Needed some numerical stability work: KPZ requires dt=0.01 with 5x substepping to avoid blowups at random parameters. KS needs T capped at 500 with NaN guards and retry logic.

To push past ARI 0.5 would probably need features targeting the EW-KPZ distinction specifically — spectral slope, structure function exponents, temporal growth rate. Or a hierarchical approach: separate the easy classes first, then tackle the hard EW-KPZ boundary.

### Exp 63: Temporal growth features

**Goal**: Break the ARI~0.5 ceiling from Exp 62 by adding 4 temporal features that target the EW-KPZ degeneracy, which spatial gradient features cannot resolve due to the stationarity theorem (Exp 54: EW and KPZ share identical stationary gradient statistics via D/ν mapping).

**New features** (computed from late-time height trajectories):
- `beta_eff`: Effective growth exponent from log-log fit of W(t)~t^β
- `vel_skew`: Skewness of multi-step velocity Δh(stride), where stride = min(10, n_avail//20). Multi-step accumulates nonlinear signal above noise floor.
- `vel_kurt`: Kurtosis of multi-step velocity
- `slope_growth`: Pearson correlation r(v, (∇h)²) — the slope-growth coupling. Bounded [-1,1], dimensionless. Directly probes the KPZ nonlinearity λ(∇h)².

**Key design decisions**:
- v1→v2 fix: Changed slope_growth from raw covariance to Pearson r. Raw covariance had BD=21.4 (~600× larger than KPZ=0.036), destroying StandardScaler normalization.
- v1→v2 fix: Changed velocity from single-step Δh (noise-dominated) to multi-step with stride~10 (accumulates λ signal above √(D·dt) noise floor). This fixed vel_skew from showing no EW-KPZ separation to a clear gap.

**Pilot (L=128, T=500, N=30/system, 180 total)**:
- Pilot improvement: HDBSCAN ARI 0.483 → 0.619 (+0.136), kNN 82% → 97.2%
- slope_growth: EW=+0.004, KPZ=+0.155 — clean separation, consistent with Exp 13
- vel_skew: EW=−0.003, KPZ=+0.113 — nonlinear drift signature detected
- EW-KPZ centroid distance: 1.07 → 2.16 (doubled)

**Full run (L=256, T=2000, N=80/system, 480 total)**:

| Metric | 6D spatial | 10D spatial+temporal | Change |
|--------|-----------|---------------------|--------|
| HDBSCAN ARI | 0.496 | 0.496 | +0.000 |
| KMeans ARI | 0.185 | 0.498 | +0.314 |
| 3-NN accuracy | 82.7% | **97.7%** | **+15.0pp** |
| EW-KPZ distance | 0.874 | 1.326 | +0.452 |

**Verdict**: Partial success — **temporal features are clearly discriminative** (kNN reaches 97.7%), but HDBSCAN does not exploit them in 10D with 480 samples. This appears to be a density-estimation limitation in this representation, not simply a lack of feature signal. KMeans ARI nearly tripled.

**Why pilot worked but full didn't for HDBSCAN**:
1. EW-KPZ feature gaps shrink at longer times: slope_growth gap 0.155→0.121, vel_skew gap 0.116→0.081. At T=2000, both systems are deeper into stationarity.
2. 480 samples in 10D gives ~1.9 effective samples/dimension — HDBSCAN density estimation is extremely noisy.
3. KS vel_skew exploded to +2.38 at T=2000 (was −0.002 at T=500), distorting standardized feature space.

**Anomalies**:
- Eden slope_growth = −0.021 (negative), while KPZ=+0.121 and BD=+0.080 (positive). Eden reaches KPZ scaling via a different dynamical path (surface nucleation vs direct λ(∇h)²), producing opposite transient coupling sign.
- KS vel_skew=+2.38 and slope_growth=+0.068 (both positive, predicted negative). At T=2000, KS instability-saturation cycles produce asymmetric velocity distributions dominated by saturation-phase 4th-order dissipation, not the bare nonlinearity.

**Predictions scorecard**: 3/7 passed (P3 beta ordering, P4 kNN>85%, P7 EW slope_growth~0). Failed: P1 ARI>0.5 (full run), P2 vel_skew gap>0.1 (0.08), P5 EW-KPZ dist>1.5 (1.33), P6 KS slope_growth<0 (+0.068).

**Post-hoc corrections (from independent review)**:

1. **The ARI ceiling is NOT only EW vs KPZ — it's a two-front problem.** HDBSCAN's 4-cluster partition is: {EW+KPZ+Eden}(240), {BD}(80), {KS}(80), {RD}(79+1noise). This means ARI is penalized on BOTH fronts: (a) EW merged with KPZ+Eden, and (b) BD split from its KPZ universality class. ARI decomposition: even with perfect EW separation, ARI would only reach ~0.73 because of the BD split. The BD-vs-continuum-KPZ multimodality is a structural problem for any density-based method — BD's discrete lattice effects create a genuinely different density region in feature space.

2. **KS beta_eff = 2.0 ± 0.0 is clipping saturation, not a measured exponent.** The code clips beta_eff to [-1, 2]. Every single KS sample hit the ceiling. The "true" KS growth exponent is unmeasured — it's just "bigger than 2." This makes beta_eff an uninformative binary flag (KS=2 vs everything else<2) rather than a continuous physical observable for KS.

3. **6D and 10D HDBSCAN partitions are identical** — not just similar ARI, literally the same label assignment (ARI=1.0 between the two). Adding 4 temporal features changed geometry (centroid distances shifted) but created zero new density separations.

4. **High kNN does NOT imply clusters exist.** kNN=97.7% indicates the features are locally discriminative, but this is compatible with a class being split into multiple tight subclusters (BD alone inside KPZ class). kNN handles multimodal classes well; density clustering does not.

5. **Eden slope_growth = −0.021 is statistically significant** (t=26.2, n=80), not estimator noise. However, the physical interpretation (different dynamical path to KPZ scaling) needs independent verification.

6. **Spectral clustering and UMAP→HDBSCAN were independently tested and also failed to beat ARI~0.50** (per external review). The ceiling is structural, not algorithmic.

7. **10D KMeans k=3 gives ARI=0.659**, the highest ARI in the entire experiment. The data naturally prefers 3 clusters: {KPZ-class+EW}(320), {KS}(80), {RD}(80). This confirms the EW-KPZ boundary and BD split are the only remaining obstacles.

**Revised key takeaway**: The ARI~0.5 ceiling has two structural causes: (a) EW-KPZ degeneracy in density space, and (b) BD multimodality within the KPZ class (discrete vs continuum growth). Breaking ARI>0.7 requires solving BOTH. Temporal features solved the discrimination problem (97.7% kNN) but not the density-clustering problem. Next steps should target the BD-continuum gap (coarse-graining features, or hierarchical clustering that handles multimodal classes).
### Exp 64: Multi-scale coarse-graining + hierarchical peel

**Goal**: Attack both structural barriers to ARI>0.5 — (a) EW-KPZ degeneracy and (b) BD splitting from KPZ class — using block coarse-graining (Exp 23 showed KPZ-BD distance drops 90%) and hierarchical peel (remove trivially-separated RD/KS first).

**Design**: Three sub-experiments, each at 4 CG scales (b=1,2,4,8):
- 64A: Clustering at each scale + multi-scale concatenation (40D = 10D × 4 scales), all 6 systems
- 64B: Peel (remove RD+KS), cluster remaining {EW, KPZ, BD, Eden} as 2-class problem
- 64C: kNN graph connectivity diagnostic — does KPZ class form 1 or 2 connected components?

**Pilot (L=128, T=500, N=30/system, 180 total)**:

Key diagnostic — kNN graph connectivity:

| Scale | KPZ class components | Sizes |
|-------|---------------------|-------|
| b=1 | **2** | {60, 30} — BD is disconnected |
| b=2 | **1** | {90} — BD merges with KPZ+Eden |
| b=4 | 1 | {90} — still connected |
| b=8 | **4** | {55,30,4,1} — everything fragments (only 16 sites) |

**Key geometric diagnostic**: at b=1, BD forms a disconnected subcluster in the 5-NN graph. A density-clustering method is unlikely to merge it with KPZ+Eden without additional inductive bias. Block CG at b=2 connects it, but b≥4 destroys feature resolution.

Clustering results — no condition exceeded ARI~0.49:

| Condition | HDBSCAN ARI | KMeans ARI | 3-NN |
|-----------|-------------|------------|------|
| All systems, b=1 (10D) | 0.487 | 0.493 | 97.8% |
| All systems, b=2 | 0.492 | 0.492 | 96.1% |
| All systems, b=4 | 0.410 | 0.493 | 90.6% |
| All systems, b=8 | 0.244 | 0.244 | 84.4% |
| Multi-scale (40D) | 0.413 | 0.493 | 92.8% |
| Peel, b=1 | −0.070 | −0.067 | 93.3% |
| Peel, multi-scale | −0.072 | −0.072 | 82.5% |

**P1 FAILED**: BD-KPZ distance INCREASED (4.06→5.47) — opposite of Exp 23. Exp 23 used only 6D spatial features. With 10D features including temporal, BD's discrete dynamics diverge under CG even as spatial features converge. The KPZ-Eden distance DID decrease (2.48→0.99), confirming CG merges continuum members but pushes discrete BD further away.

**Peel results are negative**: All peel conditions gave negative ARI (~−0.07). HDBSCAN finds 2 clusters: {EW+KPZ+Eden} vs {BD} — the same BD split. With only 2 true classes and BD mis-split, ARI goes negative. Meanwhile kNN stays at 93.3%, suggesting local information exists even when global density clustering fails.

Predictions: 2/7 passed (P2 EW-KPZ increases with CG, P5 KPZ merges at b=2).

**Verdict**: Negative result, but highly informative. The unsupervised "universality class = density cluster" hypothesis does not hold cleanly for this feature family. BD's discrete lattice effects create a geometric disconnection that coarse-graining helps spatially (b=2 connects the kNN graph) but hurts temporally (beta_eff and vel_skew diverge). The ARI~0.5 ceiling appears tied to the feature geometry, so more algorithm-swapping is unlikely to be the best next step.
