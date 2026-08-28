# Literature Audit

Date: 2026-07-04

Scope: verification and retrieval only. I fetched every source cited below in this session, or I logged the exact query and engine when the direct target was not found.

Verdict labels used here:
- FOUND-DIRECT: the searched result exists and matches the target closely.
- FOUND-ADJACENT: the searched result exists, but it is adjacent rather than an exact match.
- NOT-FOUND: the direct target was not located in the searched sources.
- NEEDS-LIBRARY: likely offline or book-only material that was not directly fetched.

## Item 1 — The exp79 lemma

Verdict: FOUND-ADJACENT

What I found: there is a real approximation-theory neighborhood around the lemma, including constrained Müntz-polynomial rates, exponential-sum approximation with exponential decay, and recent Müntz-Szász neural-network papers that explicitly connect scaling exponents to learnable power bases. I did not find a source that matches the exact bounded-coefficient, finite-window linear-target lemma as stated in the theory note.

Verified sources and quotes:
- [On the Rate of Approximation by Müntz Polynomials Satisfying Constraints](https://link.springer.com/chapter/10.1007/978-3-0348-9369-5_33) — “restrictions that would still guarantee the MUntz-Jackson rate ...”
- [Approximation of 1/x by exponential sums in [1, ∞)](https://academic.oup.com/imajna/article-abstract/25/4/685/692796) — “the error decreases like O(exp(−ck)) with the order k of the exponential sum.”
- [Discovering Scaling Exponents with Physics-Informed Müntz-Szász Networks](https://arxiv.org/abs/2601.22751) — “we prove identifiability, or unique recovery, and show that, under these conditions, the squared error between learned and true exponents scales as O(| μ − α | ² ).”
- [Radial Müntz-Szász Networks: Neural Architectures with Learnable Power Bases for Multidimensional Singularities](https://arxiv.org/abs/2602.08419) — “We formally establish this result: any C 2 function that is both radial and additively separable must be quadratic...”
- [Beckermann & Townsend, On the singular values of matrices with displacement structure](https://epubs.siam.org/doi/abs/10.1137/16M1096426) — “This paper extends the application of Zolotarev numbers to deriving bounds on the singular values of matrices with ...”

Query log, 2026-07-04:
- arXiv search: `approximation linear function exponential sums bounded coefficients rate` — no results.
- arXiv search: `distance from x to span of exponentials interval` — no results.
- Scholar: `Muntz-Jackson restricted coefficients rate theorem` — returned a constrained Müntz-polynomial chapter, not an exact bounded-coefficient rate theorem.
- Scholar: `bounded coefficients exponential sums approximation rate` — returned adjacent approximation and coefficient-constrained network papers.

Assessment for the paper: the safe wording is not “known theorem already exists,” but “adjacent approximation-theory results exist, and the exact bounded-coefficient lemma remains unlocated in the searched literature.”

## Item 2 — Optimal recovery applied to scaling/extrapolation

Verdict: FOUND-ADJACENT

What I found: the Donoho / Armstrong-Kolesár / honest-CI machinery is real and directly relevant to linear-functionals-under-convex-classes, and lattice QCD already has a strong constrained-curve-fitting / model-averaging tradition. I did not find a direct prior application to finite-size scaling, corrections-to-scaling, or continuum-extrapolation floors in the searched literature.

Verified sources and quotes:
- [Statistical estimation and optimal recovery](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-1/Statistical-Estimation-and-Optimal-Recovery/10.1214/aos/1176325367.short) — “New formulas are given for the minimax linear risk in estimating a linear functional of an unknown object from indirect data contaminated with random Gaussian noise.”
- [Optimal inference in a class of regression models](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA14434) — “We consider the problem of constructing confidence intervals (CIs) for a linear functional of a regression function ... We derive finite-sample optimal CIs and sharp efficiency bounds under normal errors with known variance.”
- [Simple and honest confidence intervals in nonparametric regression](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE1199) — “We consider the problem of constructing honest confidence intervals (CIs) for a scalar parameter of interest, such as the regression discontinuity parameter, in nonparametric regression based on kernel or local polynomial estimators.”
- [Constrained Curve Fitting](https://arxiv.org/abs/hep-lat/0110175) — “We survey techniques for constrained curve fitting, based upon Bayesian statistics, that offer significant advantages over conventional techniques used by lattice field theorists.”
- [Bayesian model averaging for analysis of lattice field theory results](https://arxiv.org/abs/2008.01069) — “Model averaging ... can incorporate systematic errors associated with model choice without being overly conservative.”

Query log, 2026-07-04:
- Scholar: `finite-size scaling minimax lower bound critical exponent estimation`.
- Scholar: `continuum extrapolation lattice QCD error bounds honest confidence interval`.
- Scholar: `finite-size scaling identifiability lower bound corrections to scaling`.
- Scholar: `lattice QCD continuum extrapolation impossibility bound`.

Assessment for the paper: the honest-CI / optimal-recovery framework is established, but I found no direct FSS corrections-to-scaling application. The manuscript wording should stay at “first application I found” or “apparently first application,” not an absolute priority claim.

## Item 3 — Reusable honest-CI software

Verdict: FOUND-ADJACENT

What I found: [RDHonest](https://rdrr.io/cran/RDHonest/man/RDHonest.html) is a real published package for honest/bias-aware confidence intervals in regression discontinuity. It exposes RD-specific smoothness and bandwidth controls, and it can compute optimal estimators via an internal modulus problem, but I did not find a public API that accepts an arbitrary user-supplied modulus or an arbitrary convex constraint class. The package [dfadjust](https://scholar.google.com/scholar?q=dfadjust+R+package) is not a fixed-length honest-CI package; it is a degrees-of-freedom adjustment utility for robust standard errors.

Verified sources and quotes:
- [RDHonest documentation](https://rdrr.io/cran/RDHonest/man/RDHonest.html) — “Calculate estimators and bias-aware CIs for the sharp or fuzzy RD parameter, or for value of the conditional mean at a point.”
- [RDHonest documentation](https://rdrr.io/cran/RDHonest/man/RDHonest.html) — `kern = "optimal"` means: “use the finite-sample optimal linear estimator under Taylor smoothness class, instead of a local linear estimator.”
- [RDHonest documentation](https://rdrr.io/cran/RDHonest/man/RDHonest.html) — `sclass` is “Smoothness class, either "T" for Taylor or "H" for Hölder class.”
- [RDHonestBME documentation](https://rdrr.io/cran/RDHonest/man/RDHonestBME.html) — “Computes honest CIs for local polynomial regression with uniform kernel in sharp RD under the assumption that the conditional mean lies in the bounded misspecification error (BME) class of functions...”
- [RDTEfficiencyBound documentation](https://rdrr.io/cran/RDHonest/man/RDTEfficiencyBound.html) — “Compute efficiency of minimax one-sided CIs at constant functions, or efficiency of two-sided fixed-length CIs at constant functions under second-order Taylor smoothness class.”
- [RDHonest documentation](https://rdrr.io/cran/RDHonest/man/RDHonest.html) — “If kern = "optimal", the "lm" object is empty, and the numeric vectors "delta" and "omega" are returned in addition. These correspond to the parameters in the modulus problem used to compute the optimal estimation weights.”
- [dfadjust package](https://scholar.google.com/scholar?q=dfadjust+R+package) — “dfadjust: Degrees of freedom adjustment for robust standard errors.”

Query log, 2026-07-04:
- Scholar: `RDHonest package confidence intervals`.
- Scholar: `dfadjust package confidence intervals`.
- Scholar: `dfadjust R package`.

Assessment for the paper: RDHonest can be used as an external correctness check for RD-style honest-CI logic, but not as a plug-in replacement for the project’s custom 7-point Gaussian design with an arbitrary bounded-exponential nuisance class.

## Item 4 — FSS / lattice identifiability prior art

Verdict: NOT-FOUND

What I found: the search results are full of standard finite-size scaling papers, corrections-to-scaling papers, and continuum-extrapolation papers, but I did not locate a paper that actually computes a Le Cam / information-theoretic / identifiability floor for corrections-to-scaling or continuum extrapolation in the lattice / kinetic-roughening / FSS setting.

Representative adjacent sources and quotes:
- [Finite-Size Effects and Irrelevant Corrections to Scaling Near the Integer Quantum Hall Transition](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.206804) — “We present a numerical finite-size scaling study of the ...”
- [Finite-size scaling at quantum transitions](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.094516) — “nonanalytic corrections due to irrelevant (bulk and boundary) ...”
- [Lattice QCD: concepts, techniques and some results](https://arxiv.org/abs/1410.3403) — “One always has to extrapolate to the continuum limit and to ...”
- [Lattice QCD at finite density](https://arxiv.org/abs/hep-lat/0610116) — “Moreover, the continuum and infinite volume extrapolations ... lower bound on the applicability range and thus also a lower ...”

Query log, 2026-07-04:
- Scholar: `finite-size scaling minimax lower bound critical exponent estimation`.
- Scholar: `continuum extrapolation lattice QCD error bounds honest confidence interval`.
- Scholar: `finite-size scaling identifiability lower bound corrections to scaling`.
- Scholar: `lattice QCD continuum extrapolation impossibility bound`.

Assessment for the paper: the claim that the project imports an established identifiability-floor analysis into FSS still looks supportable as a novelty statement, provided it is framed carefully and not as a claim about generic finite-size scaling literature.

## Item 5 — SMEFT identifiability

Verdict: FOUND-ADJACENT

What I found: SMEFT literature is rich on flat directions, profile likelihoods, and truncation uncertainty at dimension-8 and beyond. I did not find a minimax / Le Cam-style identifiability floor for Wilson-coefficient extraction under bounded higher-dimension contamination.

Verified sources and quotes:
- [Removing flat directions in standard model EFT fits: How polarized electron-ion collider data can complement the LHC](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.116002) — “Another issue that arises in global fits to the SMEFT ... dimension-8 terms which we neglect.”
- [Use and reuse of SMEFT](https://arxiv.org/abs/2009.00127) — “we suggest a widespread use of SMEFT not just as a global fitting tool ... but also as a bookkeeping framework ...”
- [Theoretical developments in the SMEFT at dimension-8 and beyond](https://arxiv.org/abs/2203.06771) — “We discuss the possibility of obtaining all-orders results in the 1 / Λ expansion for certain SMEFT observables ... and discuss the impact of these terms on experimental analyses.”
- [Profile Likelihoods on ML-Steroids](https://arxiv.org/abs/2411.00942) — “Profile likelihoods, for instance, describing global SMEFT analyses at the LHC are numerically expensive to construct and evaluate. Especially profiled likelihoods are notoriously unstable and noisy.”
- [To profile or to marginalize - A SMEFT case study](https://www.scipost.org/SciPostPhys.16.1.035?acad_field_slug=politicalscience) — Scholar snippet in this session: “We compare, for the first time, results from a profile likelihood ...”

Query log, 2026-07-04:
- Scholar: `SMEFT flat directions global fit dimension-8 truncation uncertainty`.
- Scholar: `profile likelihood degeneracy SMEFT Wilson coefficient`.

Assessment for the paper: the right framing is “adjacent literature exists on flat directions and truncation uncertainty, but I found no direct minimax identifiability-floor treatment.”

## Item 6 — Neural scaling-law fitting critiques

Verdict: FOUND-ADJACENT

What I found: there is now a substantial critique / robustness literature on scaling-law fitting, including Chinchilla replications, bias analyses, and recent geometry-based identifiability fixes. I did not find an identified-set or minimax floor treatment for scaling-law exponents under crossover / correction terms.

Verified sources and quotes:
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) — “We find that the reported estimates are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals--intervals this narrow would require over 600,000 experiments, while they likely only ran fewer than 500.”
- [Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits](https://arxiv.org/abs/2603.22339) — “Its parabolic approximation introduces systematic biases in compute-optimal allocation estimates, even on noise-free synthetic data.”
- [Practical Scaling Laws: Converting Compute into Performance in a Data-Constrained World](https://arxiv.org/abs/2605.09189) — “The dominant such scaling law form, Chinchilla's L = E + A / N α + B / D β, has three structural limitations outside that regime ...”
- [Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation](https://arxiv.org/abs/2605.08541) — Scholar snippet in this session: “we derive ... in the exponent gap from Jacobian geometry alone ... and give designs that restore identifiability ...”

Query log, 2026-07-04:
- Scholar: `Chinchilla replication scaling law critique exponent`.
- Scholar: `broken neural scaling laws identifiability exponent correction`.

Assessment for the paper: the literature supports the existence of a real fitting/robustness problem, but not yet a worst-case identifiability-floor analysis in the sense used by this project.

## Item 7 — Learned summary statistics precedent

Verdict: FOUND-ADJACENT

What I found: summary-statistics learning is well established, including IMNNs, neural approximate sufficient statistics, nuisance-hardened summaries, localization, and path-signature summaries. I did not find a paper in which summary statistics are trained against a worst-case nuisance class in a strict minimax sense.

Verified sources and quotes:
- [Automatic physical inference with information maximising neural networks](https://arxiv.org/abs/1802.03537) — “We introduce a simulation-based machine learning technique that trains artificial neural networks to find non-linear functionals of data that maximise Fisher information: information maximising neural networks (IMNNs).”
- [Neural Approximate Sufficient Statistics for Implicit Models](https://arxiv.org/abs/2010.10079) — “We consider the fundamental problem of how to automatically construct summary statistics for implicit generative models ...”
- [Nuisance hardened data compression for fast likelihood-free inference](https://academic.oup.com/mnras/article-abstract/488/4/5093/5530778) — Scholar snippet in this session: “We propose to find a reduced set of ‘nuisance hardened’ summary statistics ...”
- [Approximate Bayesian Computation with Path Signatures](https://arxiv.org/abs/2106.12555) — “We propose to use path signatures in approximate Bayesian computation to handle the sequential nature of time series.”
- [OASIS: Observation-Aware Simulation-Based Inference via Distributional Matching](https://arxiv.org/abs/2606.22572) — “Standard simulation-based inference methods often ignore this distinction ... OASIS addresses this mismatch by explicitly embedding the observation model into the simulator ...”

Query log, 2026-07-04:
- Scholar: `IMNN information-maximizing neural networks summary statistics`.
- Scholar: `semi-automatic ABC summary statistics nuisance`.
- Scholar: `nuisance-hardened summary statistics`.
- Scholar: `worst-case nuisance summary statistics neural network`.

Assessment for the paper: the closest precedent is nuisance-hardening or compression under nuisance projection, not minimax training against an adversarial nuisance class.

## Item 8 — Information-geometric RG

Verdict: FOUND-DIRECT

What I found: the Bény-Osborne information-geometric RG program exists directly, and follow-up work through 2026 continues the geometric / coarse-graining / distinguishability line. What I did not find in this line is a finite-window inference floor or an identifiability theorem for exponent recovery.

Verified sources and quotes:
- [Information-geometric approach to the renormalization group](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.022330) — Scholar snippet in this session: “We propose a general formulation of the renormalization group (RG) as a family of ... information geometry, we induce the space of Hamiltonians with a corresponding metric geometry ...”
- [Renormalisation as an inference problem](https://arxiv.org/abs/1310.3188) — “This framework then allows us to provide an information-theoretic formulation of the renormalisation group, applicable to both statistical physics and quantum field theory.”
- [Coarse-grained distinguishability of field interactions](https://quantum-journal.org/papers/q-2018-05-24-67/) — “A general way of limiting such an optimisation to certain observables is to first coarse-grain the states by a quantum channel.”
- [Information loss under coarse graining: A geometric approach](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.052112) — Scholar snippet in this session: “We can define the Fisher metric on the bare parameter space, but for coarse-grained ...”
- [Bayesian renormalization](https://iopscience.iop.org/article/10.1088/2632-2153/ad0102/meta) — Scholar snippet in this session: “scale—the distinguishability of models. Put ... RG is precisely the refined form ... induced by the Fisher metric ...”

Query log, 2026-07-04:
- Scholar: `Beny Osborne information geometry renormalization group`.
- Scholar: `information geometry renormalization group Beny Osborne`.

Assessment for the paper: the RG/info-geometry analogy is a real and published line of work, but the searched follow-up literature does not provide the finite-window floor statement that the project wants to add.

## Item 9 — Verification of the project’s existing anchor cites

Verdict: FOUND-DIRECT

What I verified:
- Donoho 1994 is real and the fetched abstract matches the theory note’s use of modulus-of-continuity language. The exact “≈ 1.25” constant is not in the fetched source; the source says “within a few percent of minimax.”
- Armstrong & Kolesár 2018 and 2020 are real and do exactly the honest / fixed-length CI construction the theory note cites.
- Lepage et al. 2001 is real and does what the theory note says about constrained curve fitting with Bayesian priors.
- Hall & Welsh 1984 was not directly fetched in this session; I only verified later papers that cite it as the tail-index minimax precursor.

Verified sources and quotes:
- [Statistical estimation and optimal recovery](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-1/Statistical-Estimation-and-Optimal-Recovery/10.1214/aos/1176325367.short) — “It is shown that affine minimax rules are within a few percent of minimax even among nonlinear rules, for a variety of loss functions.”
- [Optimal inference in a class of regression models](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA14434) — “We derive finite-sample optimal CIs and sharp efficiency bounds under normal errors with known variance.”
- [Simple and honest confidence intervals in nonparametric regression](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE1199) — “We consider the problem of constructing honest confidence intervals (CIs) for a scalar parameter of interest ...”
- [Constrained Curve Fitting](https://arxiv.org/abs/hep-lat/0110175) — “We survey techniques for constrained curve fitting, based upon Bayesian statistics ...”

Mismatch flag:
- The theory note’s placeholder “≈1.25 affine constant context” is not what the fetched Donoho abstract says. The safe statement is “within a few percent of minimax.”

Unverified list:
- Hall & Welsh (1984), original tail-index minimax paper: not directly fetched here.

## Item 10 — Quantum Darwinism × universality

Verdict: NOT-FOUND

What I found: quantum Darwinism is a real literature, but I did not find a paper tying it to a universality class or a finite-size-scaling style coarse-graining law in the searched sources.

Representative adjacent source and quote:
- [Quantum Darwinism as a Darwinian process](https://arxiv.org/abs/1001.0745) — “The Darwinian nature of Wojciech Zurek's theory of Quantum Darwinism is evaluated against the criteria of a Darwinian process as understood within Universal Darwinism.”
- [Quantum theory of the classical: Quantum jumps, Born's Rule and objective classical reality via quantum Darwinism](https://pmc.ncbi.nlm.nih.gov/articles/PMC5990654/) — Scholar snippet in this session: “To develop the theory of quantum Darwinism, we need to quantify information between fragments of the environment and the system.”

Query log, 2026-07-04:
- Scholar: `quantum Darwinism universality class`.
- Scholar: `redundancy coarse-graining pointer states scale`.

Assessment for the paper: keep this as a brief negative scan only; it does not presently support any manuscript claim.

## UNVERIFIED

- Hall & Welsh (1984) original tail-index minimax paper: not directly fetched here.
- Borwein & Erdélyi book chapters: not directly fetched here; I only verified adjacent Müntz / constrained polynomial / exponential-sum literature.

## Claims-impact table

| Project claim | Verdict label | Required rewording if any |
| --- | --- | --- |
| Lemma novelty | FOUND-ADJACENT | Replace absolute novelty language with “I found adjacent approximation-theory results, but not the exact bounded-coefficient finite-window lemma in the searched literature.” |
| “First application to corrections-to-scaling” | FOUND-ADJACENT | Soften to “first application I found” or “apparently first application.” |
| 85b CI construction can be cross-validated with existing software | FOUND-ADJACENT | Say “RDHonest can cross-check RD-style honest CIs, but there is no plug-in API for an arbitrary modulus / convex class.” |
| SMEFT gap real | FOUND-ADJACENT | Keep the gap claim, but frame it as “I found adjacent profile-likelihood / truncation / flat-direction work, not a minimax floor.” |
| Scaling-law gap real | FOUND-ADJACENT | Keep the gap claim, but frame it as “I found critique and robustness work, not an identified-set or minimax exponent floor.” |

### Additional claim notes

- The Donoho / Armstrong-Kolesár lineage is real and strong, so the manuscript should not oversell a new minimax framework. The novelty is the FSS-specific application and the explicit modulus computation, not the statistical machinery itself.
- The software story is asymmetric: RDHonest is good validation for the honest-CI logic, but it does not eliminate the need for the project’s custom construction.
- The quantum-Darwinism scan is negative enough that it should stay out of the manuscript unless a later, more targeted search finds a real universality-class link.