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

Write `theta_L=mu-delta/2`, `theta_R=mu+delta/2`, where `mu=(theta_L+theta_R)/2` is the parent common level and `delta=theta_R-theta_L` is the signed child contrast. The additional path class has its own nonnegative nuisance scale; an unknown background scale is appended when applicable. All scales are dimensionless multipliers of `lambda_ref=1e-2 mm^-1`, so physical scattering-density units are `mm^-1`. At the uniform-parent reference point, construct the **joint data Fisher** for left child, right child, other-region nuisance, and any unknown background parameter. For `c=(-1,+1,0,...)`, first require `c` to lie in the row space of the Fisher matrix; otherwise report infinite contrast variance, even if `c.T @ pinv(F) @ c` happens to be finite. For estimable contrasts, use `Var(c.T theta) = c.T F^+ c` and normal/Wald design power for a declared minimum contrast `|delta|=0.5` at two-sided alpha 0.05. Priors do not enter.

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
- `results/run-001/`: completed primary synthetic development/evaluation record. It stores configuration, seeds, per-trial sufficient statistics/metrics, source snapshot, and SHA-256 evidence. Source commit: `9106e4cadf032d46b85f5e5a0b0dd51127f60479`.

## run-001 outcome (synthetic prototype evidence only)

Protocol: 200 development-null replicates in each of four calibration strata and 200 independent train/validation replicates for each of the five scenarios. Target samples contain 4,000 tracks per replicate (3,200 in the deliberately uncovered-child case); the empty-field-calibrated background cases additionally use a separately seeded 4,000-track empty-field sample. Training, validation, null calibration, and empty-field streams use recorded disjoint seeds. The source evidence notes that the run directory itself existed as an untracked output while the runner captured whole-worktree status; the tracked source commit and per-file code hashes are exact, and the uncommitted source diff hash is empty.

| Synthetic case | Adaptive result | Comparator behavior | Interpretation |
|---|---:|---|---|
| Uniform material, known background | 6/200 splits = 3.0%; Wilson 95% CI 1.38–6.39% | Count-only splits 200/200; fixed coarse never splits; fixed fine always splits | False split is not zero; this is a measurable limitation of the prototype rule. |
| Strong left/right contrast, unknown background calibrated by independent empty field | 200/200 splits; Wilson 95% CI 98.12–100% | Count-only also splits 200/200; fixed coarse cannot represent contrast | Held-out fine-minus-coarse NLL = -46.35 (paired 95% CI -47.63 to -45.08; lower is better). |
| Right child has no paths | 0/200 splits; contrast non-estimable in 200/200 | Fixed fine is forced to split; count-only rejects from zero count | Correctly refuses unsupported local detail. |
| Weak nonzero contrast (`delta=0.2`) with 1.0 mm hit error | 0/200 splits; design power below the declared 0.80 gate for `|delta|=0.5` | Count-only splits 200/200; fixed fine worsens held-out score on average | No-split is correct for the predeclared minimum resolvable effect, but is a miss if the task is to detect every nonzero contrast. |
| Algebraically contrast-confounded nuisance (negative control, not a physical background) | 0/200 splits; contrast non-estimable in 200/200 | Count-only splits 200/200; fixed fine cannot resolve the degeneracy | Correctly rejects exact target/nuisance non-identifiability. |

All coarse/fine fits converged (0/1,000 failures). Mean individual fit times were about 0.7–5.7 ms coarse and 0.7–6.2 ms fine on the recorded runtime; the adaptive rule fits both models, so its fitting cost is approximately their sum, excluding Fisher assembly and calibration. In uniform material, mean held-out fine-minus-coarse NLL was +0.402 (paired 95% CI +0.261 to +0.543), favoring coarse; with weak contrast/high error it was +0.302 (+0.145 to +0.460), also favoring coarse. The fixed-fine model's forced detail therefore carries a predictive penalty in these cases.

The empirical null LRT cutoffs were calibrated separately by background/error stratum and are saved in `summary.json`; calibration used 200 converged null fits per stratum. The observed uniform false-split rate is 3%, below 5% as a point estimate, but its 95% confidence interval extends above 5%. Do not claim calibrated control at exactly 5%; treat this as a development estimate requiring a separately frozen calibration protocol before any confirmatory claim.

`run-001` is not Geant4 evidence, not a material-density calibration, and not independent SCI validation. The reference density `1e-2 mm^-1`, fixed 4 GeV/c, hand-specified paths, known hit-error scale, Gaussian covariance likelihood, and fixed event allocation are synthetic assumptions. The strong case also benefits from a separately supplied empty-field calibration sample; its information budget must not be confused with target-track budget.

No smoke run is an independent validation. Do not delete or overwrite any run directory.

The weak-error case has a real, nonzero child difference (`delta=0.2`) but is below the predeclared minimum effect (`0.5`). Its empirical split/no-split correctness is reported against both ground truths: “difference exists” and “difference reaches the declared resolvable scale.” This separates a missed weak physical contrast from an appropriate refusal to claim the finer scale.
