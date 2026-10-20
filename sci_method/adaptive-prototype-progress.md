# Progress ledger — identifiability-calibrated adaptive prototype

Base: `bad85e876cbb245813814b62aa0db548d615367b`; branch `codex/sci-adaptive-prototype`.

- Baseline: branch points to requested base; existing `sci_method` suite passes 23/23 tests.
- Environment: NumPy is available; SciPy is not installed. Ruling: implement the small bounded optimizer and synthetic study with NumPy/stdlib only, avoiding a new dependency; cost if wrong: optimization may be less efficient, so report iteration limits and non-convergence rather than conceal it.
- Ruling: canonical prior numeric evidence is run-007; preserve all earlier runs and only fix narrative pointers.
- Task 1: complete; regression first reproduced a non-causal downstream contribution, then passed with positive-part lever arms.
- Task 2: complete; full joint data Fisher and null-space estimability check return infinite variance for unidentifiable contrasts.
- Task 3: complete; implemented nested coarse/fine covariance fits, count-only comparator, design-power gate and stratified empirical null-LRT cutoff.
- Task 4: complete. The formal synthetic run is `run-001`: 200 evaluation replicates per scenario and 200 null calibration replicates per stratum. The adaptive rule split 6/200 uniform cases (3.0%, Wilson 95% CI 1.38–6.39%), split 200/200 strong contrasts (Wilson lower bound 98.12%), rejected the uncovered-child case in 200/200, refused the weak/high-error case in 200/200 under the declared minimum-effect criterion, and rejected algebraic contrast confounding in 200/200. All 1,000 coarse/fine trial pairs converged. The event-count rule split every covered case regardless of evidence and therefore generated false splits in uniform, weak and algebraically confounded cases. Exact metrics and held-out NLL comparisons are in the run README/CSVs.
- Formal normalization/event budget were separately declared as `lambda_ref=1e-2 mm^-1` and 4,000 synthetic target events (plus a separate empty-field calibration sample in the relevant background mode); this is not a calibrated Geant4 material density. Smoke run-001/002/003 remain retained and excluded from formal metrics.
- Task 5: canonical prior numeric source is run-007 and the paper-outline pointer is corrected. Causal regressions, Fisher/estimability tests, split/merge tests and study-design tests pass (43 total). run-001 output hashes verified. Remaining: final independent code review, final branch push; no Geant4 was run.
