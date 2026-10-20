# Identifiability-calibrated two-child split/merge prototype

**Scope:** Synthetic development prototype only. No Geant4 runs, historical CSVs, masks, oracle metrics, or SCI independent validation are used.

## Model and parameters

One parent region is divided into left/right children. The dimensionless child scales are `theta_L=lambda_L/lambda_ref` and `theta_R=lambda_R/lambda_ref`, with `lambda_ref=1e-2 mm^-1`; equivalently the parent mean is `mu=(theta_L+theta_R)/2` and signed right-minus-left contrast is `delta=theta_R-theta_L`. Physical child densities remain nonnegative. An additional path region has its own nuisance scattering scale.

The forward model uses 4 GeV/c, `z=[-400,-300,300,400] mm`, the same four-plane residual covariance, independent isotropic Gaussian hit errors, and causal Fermi–Eyges covariance kernels. All four comparators use the same target observations and Gaussian likelihood: fixed coarse, fixed fine, a count-only split rule (at least 40 tracks through each child), and an information/evidence-driven adaptive rule.

## Three background cases

1. **Known fixed:** its covariance is added as a fixed term; it is not estimated from target data.
2. **Unknown with independent empty-field calibration:** an unknown background scale is jointly estimated from target tracks and separately seeded empty-field observations that contain no target-region scattering. Empty-field residuals enter both fitting and the joint data Fisher information, but not the held-out target predictive score.
3. **Algebraic confounding negative control:** the background covariance derivative is constructed to equal the target child-contrast direction exactly. It is intentionally not claimed to be a physical background; the correct result is that the contrast is not estimable.

## Split rule and calibration

At the uniform-parent reference point, construct the **joint data Fisher** for left child, right child, other-region nuisance, and any unknown background parameter. For `c=(-1,+1,0,...)`, first require `c` to lie in the row space of the Fisher matrix; otherwise report infinite contrast variance, even if `c.T @ pinv(F) @ c` happens to be finite. For estimable contrasts, use `Var(c.T theta) = c.T F^+ c` and normal/Wald design power for a declared minimum contrast `|delta|=0.5` at two-sided alpha 0.05. Priors do not enter.

Fit the nested coarse and fine models on training observations with a nonnegative Gaussian covariance likelihood. The adaptive rule splits only if both fits converge, the contrast is estimable, predicted power is at least 0.80, and the training likelihood-ratio score exceeds the empirical higher 95th percentile calibrated from a separate uniform-material development-null set stratified by background mode and hit-error sigma. Failed calibration fits are preserved and counted; only converged null LRTs define the empirical cutoff. Calibration and evaluation seeds are disjoint. Independent synthetic evaluation residuals are used only for held-out predictive NLL and post-decision rates.

## Cases

- Uniform children: should not split often (false-split risk).
- Strong child contrast: balanced left/right path coverage, with independently calibrated unknown background.
- Right child unobserved: exact non-estimability; must reject.
- Weak contrast with `sigma_hit=1.0 mm`: test low power and missed detection.
- Algebraic background/contrast equality: explicit non-estimable negative control.

## Run records

- `results/run-smoke-001/`: exploratory pilot at run-007's `lambda_ref=1e-4 mm^-1`, 200 events, showing that the minimum design contrast had chance-level predicted power. The invocation requested 20 calibration replicates, but an early runner bug actually used 200 while config recorded 20. Preserve as an invalid pilot; do not use its rates or cutoff.
- `results/run-smoke-002/`: exploratory attempt interrupted because one coarse fit stopped at floating-point objective resolution. Its partial calibration rows are retained; no threshold or performance result is claimed.
- `results/run-smoke-003/`: small smoke run after fixing the argument handling and optimizer stopping rule, but before fixing measurement-sigma override in the high-error calibration stratum. It is a pipeline check only and is excluded from primary prototype rates.
- `results/run-001/`: planned primary synthetic development/evaluation record; configuration, seed schedule, sufficient statistics, per-trial metrics, source snapshot and hashes are stored there when run.

No smoke run is an independent validation. Do not delete or overwrite any run directory.

The weak-error case has a real, nonzero child difference (`delta=0.2`) but is below the predeclared minimum effect (`0.5`). Its empirical split/no-split correctness is reported against both ground truths: “difference exists” and “difference reaches the declared resolvable scale.” This separates a missed weak physical contrast from an appropriate refusal to claim the finer scale.
