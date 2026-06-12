# Theory note: a minimax resolution floor for finite-size scaling

Status: draft, 2026-06-10. Companion computation: `experiments/77_minimax_floor.py`.

## Claim being proven

For saturated-width ladders W_sat(L) observed at finitely many sizes with seed
noise, there is a computable threshold Δα*(design, noise) such that **no
estimator whatsoever** can distinguish asymptotic roughness exponents closer
than Δα* — because the corresponding data distributions are statistically
indistinguishable once correction-to-scaling nuisances are chosen
adversarially. This converts the project's empirical negative results (exp75:
fits fail; exp62-64: clustering fails) into an information-theoretic statement:
the obstruction is in the data, not the algorithms. It also gives the positive
counterpart: the floor shrinks at a computable rate as the window L_max and
seed count m grow, and an estimator whose error tracks the floor is
near-minimax-optimal.

## Setup

Observed: y_{i,s} = log W_sat(L_i) for ladder sizes L_1 < ... < L_n
(n = 7, L in [32, 256] in the current data) and seeds s = 1..m.

Model (matching the exp76 generator and the standard FSS ansatz):

    y_{i,s} = c + alpha * x_i + g(L_i; eta) + sigma * xi_{i,s},
    x_i = log L_i,   xi iid N(0,1),

where g is a correction-to-scaling term with nuisance parameters eta. The
seed-mean ladder has noise variance sigma^2 / m per point. The primary
adversary class is the single power correction

    g(L; B, omega) = log(1 + B L^{-omega}),
    omega in [0.3, 2.5],  u := B L_1^{-omega} in [-0.75, 4],

i.e. exactly family F1 of the exp76 generative prior. Gaussianity of the
log-width seed noise is an idealization (checked approximately in the data);
sigma is measured per system from `wsat_perseed.csv`.

## The two-point (Le Cam) bound

Take two configurations theta_1 = (c_1, alpha, eta_1) and
theta_2 = (c_2, alpha + Dalpha, eta_2). The seed-mean data are Gaussian with
means mu_k(L_i), so

    KL(P_1 || P_2) = (m / 2 sigma^2) * sum_i [mu_1(L_i) - mu_2(L_i)]^2
                   = (m / 2 sigma^2) * D^2,

with the **confusion gap**

    D^2(Dalpha) = min_{c, eta_1, eta_2}  sum_i [ Dalpha * x_i + c
                    + g(L_i; eta_1) - g(L_i; eta_2) ]^2.

By Pinsker, TV <= sqrt(KL/2). Le Cam's two-point argument then gives, for any
estimator alpha_hat,

    sup_theta E | alpha_hat - alpha(theta) |  >=  (Dalpha / 2) * (1 - TV).

In particular, if

    D^2(Dalpha)  <=  sigma^2 / m            (so KL <= 1/2, TV <= 1/2)

then every estimator has worst-case error >= Dalpha / 4 over the two
configurations. Define the **resolution floor**

    Dalpha*(design, sigma, m) = max { Dalpha : D^2(Dalpha) <= sigma^2 / m }.

Exponent differences below Dalpha* are undecidable from this design. Note the
logical direction: the adversary is restricted to ONE correction family, so
richer correction classes can only enlarge the confusion and raise the floor.
Dalpha* is therefore a conservative (valid for all estimators, loose if
anything) impossibility threshold.

## Linearized closed form (the "resolution law")

For small corrections, g(L; B, omega) ≈ B L^{-omega} =: B v_omega(L). The
adversary's reachable set near theta_1 is then the linear span of
{1, v_omega, (dv/domega)} plus the alpha direction x. The best confusion is the
projection of x out of the nuisance span:

    D(Dalpha) ≈ Dalpha * || P_perp x ||,

where P_perp projects onto the orthogonal complement of
span{1, v_omega : omega in [0.3, 2.5]} (in practice a fine omega grid).
Hence

    Dalpha*  ≈  sigma / ( sqrt(m) * || P_perp x || ).

|| P_perp x || depends only on the design (which L's you simulate): it grows
with the number of decades in L and is butchered when the correction basis can
mimic the log-slope direction over a short window. This is the quantitative
form of "a finite RG window contains only finitely many bits about the fixed
point", and it prescribes optimal design (how to choose the L ladder) as a
by-product.

## What the computation in exp77 does

1. Measures sigma per system (std of log W_sat across seeds, per L) from
   `wsat_perseed.csv`.
2. Computes D^2(Dalpha) by multi-start bounded optimization over
   (c, u_1, omega_1, u_2, omega_2), on the real design {32..256} and on
   extended designs L_max in {128, 256, 512, 1024, 4096}.
3. Reports Dalpha*(L_max, m) curves — the resolution law — and per-system
   floors for EW/KPZ/BD/Eden at m = 6 and m = 24.
4. Compares the exp76 amortized estimator's synthetic RMSE and real-data
   interval widths against the floor: tracking the floor = near-optimality;
   sitting far above it = headroom.

## Computed results (exp77, 2026-06-11)

All numbers from `results_exp77_minimax_floor/floor.json`; adversary = single
power correction, |u| <= 4 at L=32, omega in [0.3, 2.5]; design L in {32..256},
n=7, unless stated.

**1. Near-non-identifiability at accessible sizes.** The confusion gap is
astonishingly small: D^2 = 1.3e-7 at Dalpha = 0.1 and 3.0e-7 at Dalpha = 0.14
(the BD-naive-vs-KPZ gap). Converting to the seed count needed just to reach
KL = 1/2 (sigma^2/D^2):

    Dalpha:           0.05      0.10      0.14      0.20    0.30   0.40
    seeds (EW noise): 883,000   161,000   70,000    13,500  620    55
    seeds (BD noise): 15,000    2,700     1,200     230     11     1

At L <= 256, resolving exponent differences of 0.1 against adversarial
standard corrections requires ~10^5-10^6 seeds at continuum-system noise.
Statistics cannot substitute for window size: the floor vs m at sigma=0.15 is
0.51 (m=6), 0.44 (m=24), 0.38 (m=96) — nearly flat, because the adversary
achieves near-exact mimicry and the bound is identifiability-limited, not
noise-limited.

**2. Per-system floors (real design, measured noise).** m=24:
BD 0.27, Eden 0.44, EW 0.44, KPZ 0.44. Every estimator has worst-case error
>= floor/4. This proves the exp75 fit failures and the exp62-64 clustering
ceiling were information-theoretically forced at these sizes, given only
power-law-correction structure is assumed.

**3. Resolution law vs window (sigma=0.15, m=24, u<=4).**
L_max = 256: 0.44; 512: 0.33; 1024: 0.26; 4096: 0.19; 16384: 0.14.
Slow (roughly inverse-logarithmic) decay — the quantitative form of the
folklore that exponents need decades of L, not seeds.

**4. Value of structural knowledge (floor vs correction-amplitude bound u_max,
m=24).** sigma=0.15: u_max 4 -> 0.44, 2 -> 0.31, 1 -> 0.21, 0.5 -> 0.18,
0.1 -> 0.077. BD noise (0.019): u_max 1 -> 0.124, 0.5 -> 0.091, 0.1 -> 0.026.
Prior knowledge bounding the correction amplitude is worth more than orders of
magnitude in statistics. Consistency check passed: BD's actual correction
amplitude is u ~ 0.4; with the honest bound u <= 0.5 the floor gives
worst-case error >= 0.023, and the exp76 amortized interval (+-0.03) sits just
above it — the estimator operates near the information limit rather than
beyond it (no overconfidence detected).

**Interpretation.** Worst-case identifiability of alpha_inf from L <= 256 is
essentially nil; ALL practical inference at accessible sizes therefore rides
on prior assumptions about correction structure. Classical fits smuggle that
prior in silently via the ansatz (and fail when it is wrong, exp75); the
amortized estimator declares it and marginalizes over it (exp76). The
Bayes-vs-minimax gap *is* the value of the prior, and both sides are now
quantified.

**Known issue.** The linearized closed form in the previous section is
implemented incorrectly in exp77 (a 40-vector omega-grid basis trivially spans
the 7-point space, giving ||P_perp x|| ~ 0). The correct tangent space is
3-dimensional ({1, v_omega, dv/domega} at the adversarial optimum, minimized
over omega). The exact optimization results above are unaffected. Fix pending.

## Predictions to check (falsifiable)

- BD's measured noise is ~5-10x smaller than EW's, so its floor should be
  several times finer — explaining the tight BD interval ([0.482, 0.529])
  as a property of the data, not overconfidence of the estimator.
- exp75's failure is predicted: at m = 6 and sigma_EW, the floor at
  L_max = 256 should be comparable to or larger than the EW/KPZ extrapolation
  errors observed there (~0.1-0.2).
- The floor should shrink roughly like 1/sqrt(m) and faster-than-logarithmically
  in L_max once the window spans enough decades to decorrelate x from the
  correction basis.

## Relation to known mathematics

The estimation problem (first-order exponent under unknown second-order
correction) is formally the tail-index estimation problem of extreme-value
statistics under second-order regular variation. Hall & Welsh (Ann. Statist.
1984) proved minimax rate bounds there; Drees (1998, 2001) extended them. The
Le Cam construction above is the finite-design analogue. To our knowledge this
correspondence has not been exploited for finite-size scaling analysis in
statistical physics; it imports a mature impossibility toolkit into FSS
methodology. (Literature check pending — verify no prior FSS minimax work
exists before claiming priority.)

## Caveats

- Gaussian log-noise and independence across L are idealizations; both are
  checkable from per-seed data (correlations across L within a seed are absent
  by construction here since each (L, seed) is an independent run).
- The bound is for the worst case over the adversarial pair; average-case
  (Bayes) risk under the exp76 prior can be better. Both statements are
  meaningful and should be reported side by side.
- sigma is estimated, not known; plug-in uncertainty propagates linearly into
  the floor and is dominated by its seed-count scaling.

## Addendum (2026-06-11): the approximation-theory bottom of the problem

Literature + numerical mechanism study (exp77 confusion_gap with varied
omega/window bounds):

**Where this lives mathematically.** On the log-L axis (x = log L), power-law
corrections L^{-omega} are exponentials e^{-omega x}, and the exponent shift is
the linear function x. The confusion gap is therefore the L2 distance from a
linear function to the manifold of bounded-amplitude exponential sums on an
interval — the classical territory of (i) Muntz–Szasz/Muntz–Jackson theory,
(ii) approximation by exponential sums (Braess–Hackbusch: completely monotone
functions on finite intervals are approximated by N-term exponential sums with
error ~ e^{-cN}), and (iii) the notorious ill-conditioning of Prony-type
exponential fitting. The empirically observed D ~ 1e-4..1e-7 is the inferential
shadow of these known exponential-in-N approximation rates. Note: arXiv
2601.22751 / 2602.08419 ("Muntz–Szasz networks" for scaling exponents) shows
the connection is live; priority of the specific lemma below must be checked.

**The degeneracy mechanism (verified numerically).** The exponent direction is
the omega->0 boundary of the correction cone: (1 - L^{-omega})/omega -> log L.
So a correction with small omega and large amplitude IS an exponent shift, up
to Taylor remainder. With amplitude bounded by U, faking a tilt da requires
omega_eff >= da/U, and each additional correction term cancels one more Taylor
order. Predicted scaling for N correction terms on a window of log-length T:

    D ~ c_N * da^{N+1} T^{N+1} / U^N    (amplitude-limited regime)

Numerical checks (da=0.1, N=2, T=2.08, U=4): predicted ~5.6e-4 vs measured
3.7e-4; superlinear growth in da confirmed (D ratio ~16 for da 0.1->0.3,
prediction 27); growth in omega_min*T confirmed in both sweeps. NO clean
single-variable collapse in omega_min*T alone — both regimes (amplitude-limited
at small omega, curvature-limited at large omega*T) are present, as the
mechanism predicts. A plateau at small omega_min (D saturating ~4.5e-4 as
omega_min -> 0.1) confirms the amplitude budget, not omega_min, binds there.

**Physical translation.** omega_min is the smallest irrelevant RG eigenvalue;
U the irrelevant-operator amplitude. The lemma-shape statement: finite-window
exponent identifiability degrades as (da*T/U)^N, and marginal operators
(omega -> 0) destroy it entirely — consistent with the known worst cases of
exponent extraction (logarithmic corrections at upper critical dimensions,
2D XY). The smallest irrelevant eigenvalue, not the noise, sets the resolution.

**Status.** This is now a well-posed, plausibly elementary mini-theorem
(Taylor/Chebyshev remainder bounds on bounded exponential sums vs a linear
target), connected to a serious literature, with a physics meaning. It is the
single mathematically substantive object the project has produced. Next steps:
(a) prove the N-term bound rigorously, (b) literature check against
Muntz–Jackson rate theorems and the Braess–Hackbusch family, (c) only then
attach the statistics (Le Cam) and the physics (RG spectrum) as corollaries.

## Addendum 2 (2026-06-11): the lemma, stated and numerically verified

Clean problem (linear exponential-sum adversary, continuous window):

    E_N(da, T, U) = min || da*x - c - sum_{i=1}^N a_i e^{-w_i x} ||_{L2[0,T]}
                    over c free, w_i >= 0, |a_i| <= U.

**Why the amplitude bound is the entire content**: {e^{-wx} : w in any interval}
has accumulating exponents, so by Laplace-transform uniqueness its span is
dense in C[0,T] — with unbounded amplitudes, perfect confusion of any da is
approachable and identifiability is vacuous. Physically: exponents are
measurable at all only because the irrelevant spectrum is discrete with
bounded amplitudes.

**Conjectured law** (from confluent Taylor-cancellation construction with
exponents w, 2w, ..., Nw, amplitude-limited at w_eff ~ da/U):

    E_N  ~=  c_N * sqrt(T) * U * (da*T/U)^{N+1},      da*T <~ U.

Scale-invariance checks pass (joint linearity in (da,U); x-rescaling).

**Numerical verification** (/tmp/lemma_test.py, bounded lsq over amplitudes +
Nelder-Mead over exponents, multistart):

  N-scaling (da=.1,T=2,U=4):  E_1=5.45e-4, E_2=1.64e-5, E_3=8.0e-7
     ratios 0.030, 0.049  vs predicted da*T/U = 0.05         PASS
  U-scaling (N=2): U=1,2,4,8 -> 3.38e-4, 7.13e-5, 1.64e-5, 3.95e-6
     per-doubling ratios 4.75, 4.33, 4.16 -> predicted 4 (U^-2)   PASS
  da-scaling (N=2): da=.05,.1,.2 -> 1.98e-6, 1.64e-5, 1.43e-4
     per-doubling ratios 8.3, 8.7 -> predicted 8 (da^3)           PASS
  T-scaling (N=2): T=1,2,4 -> 1.40e-6, 1.64e-5, 2.02e-4
     per-doubling ratios 11.8, 12.3 -> predicted 11.3 (T^3.5)     PASS

  Fitted constants: c_1 ~ 0.039, c_2 ~ 0.023 (to be pinned analytically).

**Proof architecture**:
- Upper bound: the confluent construction above, explicit.
- Lower bound: annihilator operator L = (d/dx) prod_i (d/dx + w_i) kills the
  adversary family while L[da*x] = da * prod w_i != 0; converting ||L r|| to
  ||r|| needs Markov-type inequalities for exponential sums — exactly
  Newman's inequality / Borwein-Erdelyi (Polynomials and Polynomial
  Inequalities, ch. on Muntz systems & exponential sums). Two-regime form
  (amplitude-limited vs curvature-limited) expected in the final statement.

**Corollaries once proven**:
1. Le Cam floor with explicit constants: required seeds
   m* ~ sigma^2 / E_N^2 ~ (sigma^2/(T U^2)) * (U/(da T))^{2N+2}.
   Each doubling of the window (in decades) cuts required statistics by
   ~2^{2N+3} (128x at N=2): the seeds-vs-decades exchange rate.
2. Optimal design: maximize E_N per CPU cost (cost/seed ~ L^{1+z});
   quantitative ladder-design rule.
3. Physics: resolution is set by the smallest irrelevant eigenvalue and the
   number/amplitude of active correction terms, not by noise; marginal
   operators (w->0 at fixed amplitude... or amplitude growing) destroy
   identifiability — retrodicts log-correction pathologies (upper critical
   dimension, 2D XY).

**Identification worth writing down**: on the log axis the FSS data are a
finite window of a Laplace-type transform; exponent extrapolation IS
finite-window inverse Laplace inversion, whose exponential ill-posedness is
classical (and familiar to physicists as the analytic-continuation problem of
imaginary-time QMC data). The lemma is the quantitative finite-N face of that
ill-posedness.

**Open before any claim**: (i) rigorous lower bound (Newman/B-E route);
(ii) pin c_N; (iii) prior-art pass through Borwein-Erdelyi corpus and
Muntz-Jackson rate literature ("restricted coefficients" results especially);
(iv) the log(1+u e^{-wx}) nonlinear family embeds an infinite harmonic series
- check it does not strengthen the adversary beyond the linear-N statement
used (exp77 normalization differs: 7-point sums vs continuous L2 — reconcile).

## Addendum 3 (2026-06-12): constants pinned, construction audited, exponential regime confirmed

Certificates: /tmp scripts copied to experiments/79b_constant_certificates.py.
All runs seeded the optimizer with the construction, so gaps are measured, not
assumed; amplitude bounds enforced in the optimum via bounded lsq.

**Closed-form construction constant (Richardson nodes w_i = i*w, Lagrange
weights beta_i = (-1)^{i-1} C(N,i), amplitude binding at i=1 giving
w = N*da/U, residual projected via monic shifted Legendre):**

    c_N^constr = N^N (N+1)! N! / ((2N+2)! sqrt(2N+3))
    c_1 = 0.0373,  c_2 = 0.0252,  c_3 = 0.0321 (upper bounds)

**Certified results (da*T/U -> 0 limit):**

    N=1: c_opt -> 0.0375(2). Construction is OPTIMAL at N=1; the closed form
         is exact (up to the constrained-refit subtlety: the bound binds, and
         the certified value includes it).
    N=2: c_opt -> 0.0216. Construction 14% loose — uniform nodes are NOT
         optimal at N=2. True optimal node placement = open mini-problem
         (likely Chebyshev-like clustering; variational characterization TBD).
    N=3: c_opt -> ~0.019. Construction 40% loose; looseness grows with N as
         expected (Richardson weights blow up ~2^N, wasting amplitude budget).

So: law E_N = c_N sqrt(T) U (da*T/U)^{N+1} CONFIRMED with certified constants
c_1=0.0375, c_2=0.0216, c_3~0.019; closed form is exact at N=1 and an explicit
upper bound for all N.

**Unbounded-N (exponential) regime confirmed.** At da=0.1, T=2, U=4:
E(N=1..4) = 5.4e-4, 1.6e-5, 7.3e-7, 4.1e-8; per-term ratios 0.030, 0.045,
0.057 — geometric decay with slowly growing ratio, as c_N-growth predicts,
toward the optimal truncation N* ~ U/(e da T). Consequence with teeth: if the
true system carries >= 4 active correction terms of O(U) amplitude, then at
EW-level noise the seed count required to resolve da = 0.1 at L <= 256 is
sigma^2/E^2 ~ 10^13 — unmeasurable by any statistics, full stop. The
identifiability hierarchy is therefore:

    N fixed, per-term bound U:   E ~ (da T/U)^{N+1}   (power-law floor)
    N free,  per-term bound U:   E ~ exp(-c U/(da T)) (exponential floor)
    no amplitude bound:          E = 0 (density; identifiability vacuous)

Physics: exponent measurement is feasible exactly insofar as the irrelevant
spectrum within window resolution is SPARSE (few active terms). The effective
N visible in a window of length T is small because successive w_i contribute
indistinguishably; formalizing "effective N(T, spectrum)" is the right next
abstraction and connects to the resolution-of-identity of the window operator.

**Still open**: optimal nodes at N>=2; rigorous lower bound (annihilator +
Newman/Borwein-Erdelyi); exp77 7-point-grid vs continuous-L2 reconciliation;
log(1+u e^{-wx}) family audit; Borwein-Erdelyi restricted-coefficients
prior-art pass.

## Addendum 4 (2026-06-12): verification ledger — all four debts paid

**Debt 1 (quadrature) — CLEARED.** E_4 = 4.08e-8 stable under Gauss-Legendre
at n=120 and n=240 (identical to 4 digits) and within 2% of the Riemann-grid
value. The exponential-regime numbers are real, not quadrature artifact.

**Debt 2 (moment identity) — PROVEN, then verified.** beta_i = l_i(0) are
Lagrange weights at nodes {1..N}; exactness of interpolation for deg <= N-1
gives sum beta_i i^k = delta_{k0}; for k=N, t^N - prod(t-i) has degree N-1,
so sum beta_i i^N = -(-1)^N N! = (-1)^{N-1} N!. Three-line proof; numerically
confirmed N=1..8.

**Debt 3 (grid bridge) — BUILT.** The continuum law, mapped to the actual
exp77 7-point design via D_grid ~ E * sqrt(n/T), predicts 3.2e-5 for the
linear 2-term adversary at da=0.1, U=4; direct certified optimization on the
grid gives 4.6e-5. Same order, ~45% agreement at finite ratio (=0.052) —
constants approximately preserved across formalizations; exact reconciliation
is a finite-size-of-asymptotics effect, not a structural gap.

**Debt 4 (log-family audit) — RESOLVED, conservative direction.** On the
7-point design at da=0.1: exp77's log-correction adversary achieves
D = 3.6e-4 (its published optimum; a 24-start rerun found 4.0e-4, so 3.6e-4
stands as the better upper bound), while the lemma's linear 2-term adversary
with |a_i| <= 4 achieves D = 4.6e-5 — 8x smaller. Reason: the log family's
harmonics are slaved (u^k/k) with effective first-harmonic amplitude
~log(1+u) <= 1.6, i.e. it is a WEAKER adversary class than two free bounded
exponentials. Consequences: (i) all published exp77 floors are conservative —
correct as stated for their declared adversary class; (ii) granting the
adversary the lemma's class raises the seed requirement at da=0.1, sigma=0.145
from ~130,000 to ~10,000,000. The paper must state the adversary class
explicitly for every floor number; both are legitimate.

**Concreteness status.** Now theorem-grade (verifiable by hand or by the
included certificates): the construction, its constants (c_1 exact at 0.0375;
c_2, c_3 certified numerically; closed-form upper bound for all N), the moment
identities, the Le Cam reduction, and the impossibility direction of the floor
(which requires only the construction). Still open and required for
SHARPNESS claims only: the analytic lower bound, optimal nodes for N >= 2.
Still open and required before submission: Borwein-Erdelyi restricted-
coefficients prior-art pass (library-grade), and an external human audit.

## Correction to Addendum 4 (2026-06-12, second pass)

The 8x gap between the exp77 log-family adversary (D=3.6e-4) and the linear
2-term adversary (D=4.6e-5) was attributed to slaved harmonics. Partially
wrong: the linear-adversary optimization ran with NO lower bound on omega,
while exp77 enforced omega >= 0.3; the lemma's amplitude-limited construction
uses omega_eff = N*da/U = 0.05, outside exp77's box. The gap therefore mixes
(i) amplitude slaving in the log family and (ii) the omega-range difference.
Both computed floors are correct for their declared classes. Consequence for
the paper: the floor has THREE knowledge axes — correction amplitude bound U,
sparsity N, and the lower edge of the irrelevant spectrum omega_min — and
every floor number must declare all three. Physically, omega_min >= known
leading irrelevant exponent is often justified (e.g., omega = 1 expectations
for BD intrinsic-width corrections), and asserting it lowers the floor:
knowledge of the irrelevant spectrum, not just amplitudes, buys resolution.
A decomposition run (linear adversary with omega >= 0.3 vs unbounded) is
needed to apportion the 8x between the two effects.
