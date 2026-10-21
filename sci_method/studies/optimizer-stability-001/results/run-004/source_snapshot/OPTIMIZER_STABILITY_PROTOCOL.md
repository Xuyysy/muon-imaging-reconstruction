# Covariance optimizer numerical audit — frozen acceptance protocol

Base: `5334f3533e1ada58f0afb3e98f8de10a6f52aeb8`.
Branch: `codex/sci-optimizer-stability`. Date: 2026-10-20.
This protocol is recorded before changing the solver or comparing new fits.

## Scope and fixed panel

Reproduce the historical source snapshot in adaptive-prototype-001/run-002,
including RNG order and all generator parameters. Verify regenerated group
scatter matrices against stored sufficient statistics, not just seeds. Raw
historical observations were not saved: equality of sufficient statistics
proves likelihood-input equivalence, not the identity of every historical row.

- A: calibration known_fixed, sigma=1, replicate 54, seed 5028102055;
  uniform child levels (1,1). Fit both coarse and fine.
- B: weak_contrast_large_hit_error, replicate 173, seed 5030103731;
  original (0.9,1.1) levels. Fit both coarse and fine.
- Controls: replicate zero from all five run-002 evaluation scenarios and
  replicate zero of known_fixed sigma=1 calibration. No sample selection by
  solver outcome. Total eight original data cases / sixteen model fits.
- Analytic stress controls: zero-boundary optimum, unobserved parameter,
  equivalent duplicate kernels, near-singular positive covariance, invalid
  covariance, forced iteration exhaustion, and a high-information scalar
  model where a small update still leaves an appreciable KKT residual.

No new statistical power study, null recalibration, low-budget grid or Geant4.
Keep all old result files unchanged. This is solver development evidence.

## Objective, scales and acceptance

Parameters are the existing dimensionless density scales. Use fixed coordinate
scales s_i=max(1, initial_i); do not change scales with the solution to make a
residual pass. Let r_i=g_i for theta_i>0 and min(g_i,0) at an exact zero bound.
The KKT residual is max_i |s_i r_i| for the full summed NLL (no N normalization).

1. Required first-order residual <= 1e-6, explicitly on this fixed-scale KKT
   residual. The value is a numerical stationarity specification, not a
   replacement for the scientific power or LRT threshold. A tiny line-search
   step is never sufficient. Feasibility theta>=0 is exact by projection.
2. Independent objective agreement:
   |f(candidate)-f(reference)| <= max(1e-8,512 eps max(1,|f|)).
   The 1e-8 floor is the declared absolute NLL accuracy requirement; the second
   term allows accumulated binary64 matrix arithmetic. This is a practical
   audit tolerance, not a rigorous global optimality certificate.
3. For locally identifiable, well-conditioned solutions, parameter agreement
   max_i |theta_i-theta_ref_i|/max(1,|theta_ref_i|) <= 1e-5.
   For singular models compare predicted covariances (relative Frobenius
   error <=1e-6) and KKT instead; parameter equality is not identifiable.
4. Gradient: independent central differences at h=s_i eps^(1/3) times
   {0.1,1,10}, one-sided second-order differences at zero, with best-step
   component discrepancy <= 5e-5 + 2e-5 |g_i|. Record every h and error; this
   permits cancellation in total NLLs of order 1e4 while resolving the score.
   Also cross-check analytic scores with an independently coded inverse/trace
   formula. Analytic stress controls have known optima.
5. Check covariance eigenvalues/condition numbers and observed local Hessian.
   KKT is necessary, not proof of a global optimum. Use independent bounded
   SciPy optimization plus free-variable root polishing and multiple starts
   for A/B. Solver `success` alone is not an acceptance condition.
6. Numerical status: converged (KKT met); precision-limited (no resolvable
   decrease, KKT unmet, must retain independent evidence separately);
   unproven (iteration budget/other stall); failed (invalid/nonfinite model).
   The compatibility Boolean is true only for converged fits. No tolerance
   relaxation or seed replacement is allowed to make a failure disappear.
7. Compare LRTs at the original frozen thresholds. A provisional numerical
   decision interval is raw LRT ± 2(error_coarse+error_fine), where each error
   uses the objective agreement budget and observed reference discrepancy.
   This is a diagnostic uncertainty allowance, not a statistical confidence
   interval or a rigorous optimization-error bound. Flag threshold overlap
   and scores below this resolution; do not change alpha or the cutoff.

## Implementation and evidence plan

1. Save exact legacy traces (including all rejected line-search candidates),
   initial/final full NLL, gradients, diagonal Fisher scales and KKT residuals.
2. Investigate stable covariance-difference evaluation and score arithmetic.
   Implement a small solver change only after identifying the failure;
   preserve the tuple API and expose detailed statuses through an added API.
3. Test real numerical behavior before implementing changed convergence logic.
4. Compare the fixed panel with a separate SciPy method; record versions,
   BLAS/platform, source snapshots, input hashes, timings and all failures.
   SciPy is audit-only unless evidence justifies otherwise.
5. Record the six-cell execution command for review but do not run it.
   A go decision requires both known failures resolved, no false-success stress
   regression, acceptable reference agreement, and an explicit disposition of
   any threshold-level LRT ambiguity.

Reference: SciPy's [L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
distinguishes objective-change and projected-gradient termination; either
solver's return flag must be checked against the criteria above.
