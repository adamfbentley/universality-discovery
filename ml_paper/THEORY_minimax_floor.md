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

For each admissible pair satisfying this condition, the construction gives the
stated lower bound on worst-case expected absolute error over that pair. This is
not an absolute claim that every exponent difference below Dalpha* is
undistinguishable. The result is conditional on the Gaussian observation model,
the finite design, the `W_sat` summary, and the bounded correction class. The
adversary is restricted to one correction family; enlarging that nuisance class
cannot weaken this particular lower-bound construction.

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
>= floor/4 for the admissible two-point constructions in the specified model.
This does not prove that the particular exp75 fits had to fail, and it does not
establish that the exp62-64 clustering ceiling was inevitable: those clustering
experiments used richer observables than the `W_sat` ladder bounded here.

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

**Interpretation.** Within this model, observable, design, and nuisance class,
worst-case identification of alpha_inf from L <= 256 is weak, so practical
inference depends strongly on assumptions about correction structure.
Classical fits encode those assumptions through an ansatz; the amortized
estimator declares and marginalizes over a prior (exp76). The computed
Bayes-versus-minimax gap quantifies the value of that prior only within the
declared setup.

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

## Appendix A (2026-06-13): full derivation of the construction and constant

Goal: approximate da*x on [0,T] by c + sum_{i=1}^N a_i e^{-w_i x} with
|a_i| <= U; derive the upper bound E_N <= c_N sqrt(T) U (da T/U)^{N+1}.

Step 1 (building blocks). For nu > 0 define f_nu(x) = (1 - e^{-nu x})/nu.
Taylor: f_nu(x) = x - nu x^2/2! + nu^2 x^3/3! - ... Each f_nu is an
admissible combination of a constant and one exponential with amplitude
1/nu (times whatever overall coefficient is applied).

Step 2 (confluent nodes, Richardson weights). Take nu_i = i*w for
i = 1..N and weights beta_i with sum_i beta_i = 1 and
sum_i beta_i i^k = 0 for k = 1..N-1. These are the Lagrange
extrapolation-to-zero weights beta_i = l_i(0) on nodes {1..N}; explicitly
beta_i = (-1)^{i-1} C(N,i). Proof of the moment identities: for any
polynomial p of degree <= N-1, interpolation at N nodes is exact, so
sum beta_i p(i) = p(0); take p = t^k. For k = N: t^N - prod_j(t-j) has
degree N-1, so sum beta_i i^N = [t^N - prod(t-j)]_{t=0} = -(-1)^N N!
= (-1)^{N-1} N!. (Verified numerically N = 1..8.)

Step 3 (residual). g(x) = da * sum_i beta_i f_{i w}(x) is admissible and

  da*x - g(x) = da * sum_{k>=N} (-1)^k w^k x^{k+1}/(k+1)! * M_k,
  M_k = sum_i beta_i i^k,  M_N = (-1)^{N-1} N!,

so the leading residual is -da * w^N x^{N+1}/(N+1) + O(w^{N+1} T^{N+2}).

Step 4 (projection). Re-optimizing all amplitudes and the constant
projects the residual onto the orthogonal complement of the family's
tangent space, which contains polynomials of degree <= N (from
perturbing beta and c). The L2[0,T] distance of x^{N+1} from P_N is the
norm of the monic shifted Legendre polynomial of degree m = N+1:
dist = T^{N+1} sqrt(T) * m!^2 / ((2m)! sqrt(2m+1)).

Step 5 (amplitude binding). The construction's amplitudes are
a_i = da * beta_i/(i w); the largest is |a_1| = N da / w. The bound
|a_1| <= U forces w >= N da / U; insert w = N da/U into Step 3-4:

  E_N <= [N^N (N+1)! N! / ((2N+2)! sqrt(2N+3))] sqrt(T) U (da T/U)^{N+1}.

Values: c_1 = 0.0373, c_2 = 0.0252, c_3 = 0.0321. Certified against
direct optimization: exact at N=1 (0.0375 +- grid error); loose by 14%
(N=2, true 0.0216) and ~40% (N=3, ~0.019) because uniform nodes i*w are
suboptimal beyond N=1 (optimal node geometry: open).

Step 6 (vacuity, for completeness). If amplitudes are unbounded the
infimum is 0: the span of {e^{-w x} : w in any interval} is dense in
C[0,T], because a finite measure annihilating it has a Laplace transform
analytic in w and vanishing on a set with accumulation points, hence
identically zero; Hahn-Banach then gives density. So every floor
statement is conditional on the amplitude bound — that is the content,
not a technicality.

## Appendix B (2026-06-13): corollaries not yet recorded elsewhere

B1. RG equivariance of the floor. The model class maps to itself under
L -> bL with alpha fixed and u -> u b^{-omega}; therefore
floor([bL1, bL2], U) = floor([L1, L2], U b^{-omega}). The window length
in decades T is invariant; only the effective amplitude flows. The floor
obeys the same flow as the corrections; verified directly on the law of
Appendix A (substitute U -> U b^{-omega}).

B2. Compute allocation. Moving the window up one factor of b multiplies
the confusion gap by ~b^{N omega} (smaller corrections, easier
discrimination) while per-seed cost grows like b^{d+z}. Comparing Fisher
information per unit cost: pushing to larger L wins iff 2 N omega > d+z.
For 1D KPZ-class growth (omega ~ 1, d+z = 2.5, N = 1 dominant
correction): 2 < 2.5 — marginal; neither brute statistics nor brute size
dominates, which matches the project's experience. (Heuristic until the
sharp lower bound is proven; uses the certified upper-bound law.)

B3. Identification with profiled Fisher information. For Gaussian noise,
KL between nearby models = quadratic form of the Fisher metric, so the
confusion-gap minimization is exactly the profiled Fisher information of
alpha (nuisances = correction parameters), and the floor is the global
(Le Cam) version of the profiled Cramer-Rao bound. The exponential
ill-conditioning of the model manifold is the "sloppiness" of
Machta-Transtrum-Sethna parameter-space compression; this work computes
its consequence for the one stiff parameter physicists fit.

## Appendix C (2026-06-13): positioning map (literature anchors found so far)

Each fragment of this work has a mature home; the composition appears
unclaimed. To cite and check before submission:

- Declared priors on correction amplitudes: standard in lattice QCD since
  Lepage et al., "Constrained curve fitting" (hep-lat/0110175): priors on
  excited-state amplitudes when fitting C(t) = sum A_n e^{-E_n t} — the
  same structure as our problem. exp76 is this philosophy, amortized,
  with a prior over correction families, applied to FSS.
- Fit-form systematics by model averaging: Jay & Neil (arXiv:2008.01069),
  Neil & Sitison ICs — selects/averages among forms; does not bound all
  estimators. The floor is complementary, not competing.
- Partial identification: Manski (set identification; "law of decreasing
  credibility"); Imbens-Manski intervals. Our zero-noise floor is the
  diameter of the identified set; the knowledge axes (N, U, omega_min)
  are an assumption hierarchy in his sense.
- Conditional stability / regularization on compact classes: Tikhonov;
  Isakov. Uniform decidability on a bounded class, none absolutely, is
  the standard structure of ill-posed problems.
- Analytic continuation bounds: two-constants theorem (harmonic measure);
  Demanet-Townsend stable extrapolation; Trefethen. Note: physics
  corrections are generically asymptotic, not analytic at 1/L = 0, so
  these apply to the fitted model class, not to nature directly.
- Exponential-sum approximation: Braess-Hackbusch (e^{-cN} rates on
  finite intervals); Prony ill-conditioning; super-resolution separation
  conditions (Candes-Fernandez-Granda; Moitra; Batenkov school) — the
  omega -> 0 closure degeneracy is their merging-frequencies regime.
- Singular-value decay of structured matrices: Zolotarev numbers
  (Beckermann-Townsend) — the likely sharp source of the exponential
  regime; check whether the amplitude-constrained two-regime law is a
  corollary.
- Sloppy models: Machta, Chachra, Transtrum, Sethna (Science 2013) —
  RG-irrelevant directions are sloppy FIM directions; this work is the
  estimation-theoretic side of that program for exponent recovery.
- Searched and not found (web-grade, 2026-06): minimax/identifiability
  floors for corrections-to-scaling or continuum extrapolation in the
  FSS, kinetic-roughening, or lattice literature. Library-grade pass on
  Borwein-Erdelyi (restricted-coefficient Muntz results) still owed.

Status note: the derivation chain is now fully recorded in this file
(setup -> Le Cam -> construction -> constants -> certificates ->
corollaries -> positioning). Open items unchanged: analytic lower bound,
optimal nodes N >= 2, decomposition of the omega-range vs amplitude
effects (Correction to Addendum 4), external audit, library passes.

## Appendix D (2026-06-13): closed form of the construction; reduction to
## polynomial approximation of the logarithm

D1. The construction in one line. With Richardson weights
beta_i = (-1)^{i-1} C(N,i) on harmonic nodes w, 2w, ..., Nw, the binomial
theorem gives sum_i beta_i t^i = 1 - (1-t)^N, so the approximant's slope is

    g'(x) = da * [1 - (1 - e^{-w x})^N],

and the residual slope is exactly

    r'(x) = da * (1 - e^{-w x})^N  <=  da * (w x)^N.

This replaces the Lagrange/moment machinery of Appendix A (kept for the
record): integrating r' and applying the amplitude binding w = N da / U
reproduces the law E_N <= c_N sqrt(T) U (da T/U)^{N+1} in a few lines.
Interpretation: N correction terms conspire optimally (within harmonic
nodes) as the N-th power of the single-term disguise.

D2. Substitution t = e^{-w x}. Exponential sums on harmonic nodes are
polynomials of degree N in t; the tilt da*x becomes -(da/w) log t. The
uniform-node confusion problem is therefore EXACTLY:

    best approximation of log(1/t) on [e^{-w T}, 1]
    by degree-N polynomials with coefficient bounds.

Consequences:
- The exponential-regime constant (E_min ~ exp(-c U/(da T)) for free N)
  becomes computable in closed form from the Bernstein-ellipse rate for
  polynomial approximation of log on [t0, 1], t0 = e^{-w T}. Previously
  this constant was heuristic.
- The analytic lower bound, restricted to harmonic nodes, reduces to
  classical constrained polynomial approximation of the logarithm, where
  Chebyshev/Markov-type constants are exactly known. This is now the
  recommended proof route. The general-node lower bound (and optimal
  nodes for N >= 2) remains open; general nodes do not reduce to
  polynomials.

D3. Why the exponent (N+1) in the law is protected. The nonlinear
family's full tangent space (free exponents) contains x e^{-w_i x}
directions, which in the confluent limit reach polynomials beyond P_N
and would naively cancel the leading residual term. They do not, because
the amplitude constraint binds at the optimum (|a_1| = U) and freezes
those directions. This is consistent with (and explains) the certified
result that the optimizer improves the CONSTANT over the construction
(14% at N=2, ~40% at N=3) but never the EXPONENT, which was verified
independently in all four variables (Addendum 2).

Verification status: D1 is exact algebra (checked at N=1,2 by hand; the
binomial identity is elementary). D2 is a change of variables, exact for
harmonic nodes. D3 is an observation consistent with all certified
numerics; the KKT characterization of the constrained optimum (which
constraints bind for general N) is open and is the route to exact c_N.

## Appendix E (2026-06-13): the floor is observable-agnostic (normal form)

Referee item: is the floor specific to W_sat(L) ladders? No. Computation in
experiments/80_floor_generality.py (results_exp80_floor_generality/).

The reduction is the content. Any observable O ~ X^theta * (1 + sum
b_k X^{-w_k}) in a control variable X, written on x = log X, becomes a tilt
theta*x plus bounded decaying exponentials sum a_i e^{-w_i x} — the SAME
object the floor is computed for. Consequences, stated honestly:

- A (roughness alpha, control L) and B (growth beta, control t) over the same
  number of decades give an IDENTICAL confusion gap (4.65e-5, ratio 1.00).
  This is not independent empirical evidence — it is the normal form made
  visible: in log-control coordinates A and B are the same computation, so no
  observable-specific factor enters. The floor is a function of (window in
  decades, correction spectrum, target precision) alone. That A and B coincide
  exactly is the cleanest possible statement of observable-agnosticism, but it
  is structural, not a separate confirmation; we present it as such.

- C (lattice-style correlator C(t)=A e^{-E0 t}(1+sum b_k e^{-dE_k t}),
  resolving the ground-state energy E0 under excited-state corrections) is a
  genuinely different instance: the signal is linear in t (not log t) and the
  corrections are exponentials in t directly. It exhibits the same exponential
  ill-posedness (gap 8.4e-4 for dE0=0.05 on an 11-slice Euclidean window).
  This is exactly the ground-state vs excited-state extraction tension lattice
  QCD manages daily with Lepage-style amplitude priors — i.e. a second domain,
  with a different ansatz, in which the same floor mechanism operates.

Takeaway for the paper: the floor is not a property of surface-growth
roughness ladders; it is a property of leading-exponent estimation under
unknown decaying corrections, a normal form shared by roughness exponents,
growth exponents, correlation-length fits, and lattice energy extraction.
The amplitude bound (prior) and the window length in decades are the only
inputs that matter; the physical identity of the exponent does not.

Open: a fully worked second-domain *estimator* demonstration (not just the
floor) on real correlator data would further strengthen this; here we
establish the floor's generality analytically and confirm it numerically in
three settings.

## Appendix F (2026-07-02): the observable-information hierarchy — PROPOSED
## PROGRAM (exp81). Definitions and closed forms only; NO RESULTS YET.

F0. The scope condition, stated precisely. Every floor in this note bounds
estimators that are measurable functions of a DECLARED observable (the
log-W_sat ladder, or the exp80 analogues). The bound is in TV distance
between the data distributions of that observable, so it is estimator-
agnostic GIVEN the observable, but says nothing about estimators consuming
richer data. "No architecture can beat the floor" is true per-observable and
false as an unconditional statement. The hierarchy of observables —

    Level 0: single-summary ladder
    Level 1: K-channel summary ladders
    Level 2: full spectrum / two-point statistics
    Level 3: raw configurations

— therefore carries a hierarchy of floors, and the RATIO of floors between
levels, as a function of the declared correction class, is a well-posed
quantity. Exp81 computes it for a family where every level is exactly
tractable. (Framing note for the paper: this turns the referee objection
"try a transformer on raw fields" into a computable question, and it is the
precise content behind the informal trichotomy model-limited /
optimization-limited / information-limited.)

F1. Multivariate (Level-1) confusion gap. Estimator observes y_{i,s} ∈ R^K,

    y_k(L_i) = c_k + θ_k(α)·x_i + Σ_j u_{k,j} e^{−ω_j x_i} + noise,

noise ~ N(0, Σ) across channels (Σ measured from seeds), independent across
L and seeds; θ_k(α) known per-channel exponent maps. Define

    D²(Δα) = min_{c, {ω_j} SHARED, |u_{k,j}| ≤ U_k} Σ_i Δμ(L_i)ᵀ Σ⁻¹ Δμ(L_i),

floor as before. The physics question becomes algebra: if each channel has
its own private corrections, the adversary defeats channels one at a time
(expected gain ~√K at best); if the correction spectrum {ω_j} is shared
(same irrelevant operators feeding every observable), consistency across
channels over-determines the nuisance and the gain can be much larger.
Whether richer summaries help is decided by correction SHARING and noise
CORRELATION, not by K. This is exp81 Part A; hypotheses H-A1–H-A3 and gates
in EXP81_PLAN.md.

F2. Exactly solvable testbed for Levels 2–3: stationary fractional EW.
1D Gaussian field, independent Fourier modes with spectrum

    S(k) = D / ( ν|k|^z + ν₂|k|^{z+ω̃} ),

so that α = (z−1)/2 (tunable), and W²(L) = A·L^{2α}(1 + b·L^{−ω_eff} + ...)
with b ∝ ν₂/ν. Mode-sum caveat: ω_eff = ω̃ only for ω̃ < 2α (IR-dominated
correction integral); for ω̃ > 2α the correction is UV-dominated and ω_eff
saturates at 2α. The realizable correction range at α ≈ 0.5 is ω_eff ∈ (0, 1],
and ω_eff must be measured, not assumed. The case ω̃ = 2−z is the
near-marginal stress case at α ≈ 0.5. Because the
family is Gaussian: (i) sampling is exact (draw modes, inverse FFT);
(ii) raw-field KL between two parameter settings is a mode sum,
KL = Σ_k [S₁/S₂ − 1 + ln(S₂/S₁)] per complex mode (conventions to be
verified numerically against sampled likelihoods before use); (iii) exact
Fisher information of α from raw fields is analytic,
I(α) = Σ_k (∂ ln S/∂α)², ∂ ln S/∂α = −2 ln|k| · ν|k|^z/(ν|k|^z + ν₂|k|^{z+ω̃}).
Hence the Level-3 floor is an explicit finite-dimensional optimization (Le
Cam with the adversary inside the family), with no simulation noise in the
theory, and any trained estimator has a computable ceiling it cannot beat —
a self-auditing property that catches leakage bugs automatically.

F3. What the hierarchy can and cannot conclude. The testbed is Gaussian, so
the spectrum is sufficient and Level 2 = Level 3 exactly there; the
interesting measured gap is Level 0 vs Level 2/3 under adversarial
corrections. Nothing in F2 bounds estimators on NON-Gaussian raw fields
(BD, KPZ configurations) — that requires either non-Gaussian likelihood
bounds or empirical two-sample methods, and is deliberately deferred (exp82+)
until the methodology is validated where the answer is checkable.

F4. Status. PROPOSED. All numbered claims above are definitions or standard
identities; the only physics content asserted (α = (z−1)/2, correction form,
KL/FIM conventions) carries explicit numerical verification gates
(G-B1–G-B5, EXP81_PLAN.md) that must pass before any number from this
program is cited.

## Appendix G (2026-07-03): the Donoho mapping — this floor is a modulus of
## continuity, and what follows from that

Source: adversarial statistical review (archive/ai_execution/SIMULATED_AUDIT.md); mapping to be
confirmed by the human audit (archive/ai_execution/AUDIT_BRIEF.md Q5), but the structural match is
exact and is recorded here.

G1. The mapping. Let M = {μ(c,α,η) = c·1 + α·x + g(·;η) : η ∈ class,
c ∈ R, α ∈ A} ⊂ R⁷ be the set of achievable mean vectors, and T the target
"functional" α. Define

    D(Δα) = inf { ‖μ₁ − μ₂‖₂ : μᵢ ∈ M, α₂ − α₁ = Δα }        (this note)
    ω(ε)  = sup { α₂ − α₁ : μᵢ ∈ M, ‖μ₁ − μ₂‖₂ ≤ ε }          (Donoho 1994)

These are inverse to each other, and the resolution floor is exactly
Δα* = ω(σ/√m). For CONVEX M under Gaussian noise, Donoho ("Statistical
Estimation and Optimal Recovery", Ann. Statist. 22, 1994, building on
Ibragimov–Khasminskii) characterizes the minimax risk of estimating a linear
functional by this modulus, with minimax AFFINE estimators within a small
explicit factor (≈1.25 — constant to be confirmed at the primary source) of
fully minimax. Armstrong & Kolesár (Econometrica 86, 2018) give the applied
form — fixed-design regression, Gaussian errors, linear functional, convex
nuisance class, honest confidence intervals — in active econometric use.

G2. The identifiability nuance (why T needs care). T is a function of the
parameters, not automatically of μ: if two parameter settings produced the
same μ with different α, T would be ill-defined on M and ω(0) > 0. The ω→0
correction degeneracy (Addendum 1) is precisely the near-violation of this:
ω(0) = 0 holds on the bounded class (no exact mimicry), but ω(ε) is large at
small ε. In Donoho's language the entire project result reads: *for FSS
designs at L ≤ 256, the modulus of the exponent functional over the
bounded-correction class is enormous at the noise scale.* The exp79 lemma is
the closed form of this modulus for exponential-sum classes.

G3. Convexity status. Our correction manifold {g(·;u,ω)} is nonconvex (a
2-parameter curved family), so Donoho sharpness does not apply verbatim.
Over the convex hull: floors can only increase (larger adversary ⇒
conservative impossibility, same direction as all our approximations), and
sharpness holds relative to the hull. The computable question — how much
larger is the hull floor? — is exp83 task 2.

G4. Consequences for claims (binding for the paper):
- The phrase "first minimax bound for FSS" is retired. Correct statement:
  first application of the linear-functional minimax/optimal-recovery
  machinery to corrections-to-scaling, with (i) a physically motivated
  adversary class and computed floors for real designs, (ii) a closed-form
  modulus for bounded exponential-sum classes (exp79) with an RG dictionary,
  (iii) a declared-prior amortized estimator as the constructive counterpart.
- Cite: Donoho 1994; Donoho–Liu; Ibragimov–Khasminskii; Cai–Low; Armstrong–
  Kolesár 2018 (+ software); alongside the existing Hall–Welsh/Drees and
  approximation-theory anchors.
- The sharpness program (analytic lower bound on E_N) is demoted from
  blocking to enrichment: pursue hull-sharpness via G1 first; keep the
  annihilator/Newman route as the mathematically interesting appendix.

G5. Independent verification (2026-07-03, this session). Two checks performed
from this note's formulas alone, without reference to exp77's code:
(i) D²(Δα) monotone increasing over Δα ∈ [0.02, 0.6] on the real design
(11-point scan, multistart L-BFGS-B) — the floor definition's implicit
assumption holds; a running-max guard should still be added to the code
(exp83 task 1). (ii) Independent reimplementation reproduces the BD m=24
floor: 0.271 vs the published 0.27, and D²(0.1) = 1.66e-7 vs the note's
1.3e-7 (both are upper bounds by construction; the discrepancy direction is
consistent). This is the first fully independent replication of the
project's headline number.
