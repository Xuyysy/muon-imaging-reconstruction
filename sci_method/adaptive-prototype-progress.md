# Progress ledger — identifiability-calibrated adaptive prototype

Base: `bad85e876cbb245813814b62aa0db548d615367b`; branch `codex/sci-adaptive-prototype`.

- Baseline: branch points to requested base; existing `sci_method` suite passes 23/23 tests.
- Environment: NumPy is available; SciPy is not installed. Ruling: implement the small bounded optimizer and synthetic study with NumPy/stdlib only, avoiding a new dependency; cost if wrong: optimization may be less efficient, so report iteration limits and non-convergence rather than conceal it.
- Ruling: canonical prior numeric evidence is run-007; preserve all earlier runs and only fix narrative pointers.
- Task 1: complete; regression first reproduced a non-causal downstream contribution, then passed with positive-part lever arms.
- Task 2: complete; full joint data Fisher and null-space estimability check return infinite variance for unidentifiable contrasts.
- Task 3: complete; implemented nested coarse/fine covariance fits, count-only comparator, design-power gate and stratified empirical null-LRT cutoff.
- Task 4: implemented; smoke runs exposed inadequate information at run-007's 200-event/reference scale, an optimizer stopping issue, and a sigma-stratum calibration-generation mismatch. All are preserved in their own run directories and excluded from primary rates. The formal scale/budget were separately declared as `lambda_ref=1e-2 mm^-1` and 4,000 synthetic events, with no claim this is a calibrated Geant4 material density.
- Task 5: run-007 is the canonical prior numeric source; outline pointer corrected. Full post-run structural/hash/review checks and branch delivery remain.
