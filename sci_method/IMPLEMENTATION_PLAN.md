# SCI forward-model prototype implementation plan

## Scope and scientific boundaries

Implement only a deterministic, small-angle, fixed-4-GeV numerical forward model and hand-specified synthetic identifiability study. Geometry is the four nominal detector centers `(-400, -300, 300, 400) mm`; hit x/y errors are independent Gaussian with an explicit common sigma. No Geant4, historical result mutation, truth-mask/oracle input, full reconstruction optimizer, adaptive split/merge rule, or performance claim.

Identifiability means data-likelihood information about local parameter contrasts. Prior/regularization precision is reported separately and never counted as observation information. Synthetic scenarios and tolerances are fixed in tests before running the study.

## Work items

1. Add failing tests for strict event IDs/composite run keys and the forward-model covariance, limits, likelihood, gradient, and Fisher behavior.
2. Implement the linear hit-to-residual map, detector covariance, exact slab-clipped Fermi–Eyges path integrals, Gaussian NLL, and analytic gradient using NumPy only.
3. Implement Fisher eigensystem and nuisance-profiled contrast information, keeping prior precision separate.
4. Add predeclared identifiable and non-identifiable synthetic path scenarios, fixed seed, immutable unique output directory, config/input/code hashes, CSV/JSON summaries, and reproducibility note.
5. Correct the literature matrix (arXiv version identity; Wang/Schultz/Qi 2009 papers) and scientific language on identifiability; add full derivation and scope/limitations.
6. Run focused tests and study in the bundled scientific Python; run available legacy regression tests; inspect outputs and repository diff; commit and push only this branch.

## Acceptance

- Analytic detector covariance agrees with fixed-seed Monte Carlo within predeclared 2% relative tolerance for the tested nonzero entries.
- Covariances symmetric PSD; valid likelihood covariance SPD; zero scatter and zero hit-noise limits are explicit.
- NLL and gradient finite and gradient agrees with central finite differences within predeclared tolerance.
- A constructed exact null contrast is reported as a zero-information eigenmode; regularization changes only prior/posterior precision, not data Fisher.
- Contract rejects fractional and boolean event IDs and requires run ID; model package has no truth/mask/oracle API.
- Outputs are created only under a new study-specific path and existing paths are never overwritten.
- No Geant4 run or large parameter/quality scan.
