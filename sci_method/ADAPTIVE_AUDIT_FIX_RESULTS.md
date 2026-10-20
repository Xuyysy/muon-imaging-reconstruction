# SCI adaptive prototype audit-fix results

**Audit branch:** `codex/sci-adaptive-audit-fix`
**Scientific base:** `4a179a753fc1db5e5d488c43831e3edfa4afac7e`
**Scope:** algebraic-confounding covariance parity, run-001 rule ablations, a corrected independent synthetic run, finite-precision diagnosis, and a proposed Geant4 development diagnostic. No Geant4 was run and no historical CSV or `main` content was changed.

## 1. Algebraic negative-control correction

The target generator iterated in insertion order `left`, `right`, `other`. It assigned `kcontrast` in the left and right branches but not in the `other` branch, so `other` reused the right-child value from the previous iteration. The generator therefore included `background_level * 0.5 * K_right` in the other path class, while the likelihood correctly specified a zero algebraic-background derivative there.

The shared helper `_algebraic_confounded_background_kernel(region)` now returns `-0.5 K_left`, `+0.5 K_right`, and an explicit zero matrix for `other`; both generator covariance assembly and likelihood derivatives use that helper. `_synthetic_target_covariance` centralizes the generator covariance. A regression test reconstructs all three likelihood covariance matrices at the declared generating parameters and asserts equality at `1e-12` tolerance. The test was observed failing before the fix and passing after.

The original `adaptive-prototype-001/results/run-001/` has not been overwritten. A fresh, independently seeded corrected full study is recorded at [`adaptive-prototype-001/results/run-002/`](studies/adaptive-prototype-001/results/run-002/), with source commit `1797120ebc4904ee92360fef2ccebe66bd88ed9f`.

| Algebraic negative-control statistic | Original run-001 (generator mismatch) | Corrected run-002 (generator and likelihood agree) |
|---|---:|---:|
| LRT-only selections | 21/200 (10.5%) | 6/200 (3.0%) |
| Fisher + power + calibrated LRT selections | 0/200 | 0/200 |
| Mean raw LRT | `1.96e-11` | `1.37e-11` |
| Maximum raw LRT | `1.71e-10` | `1.35e-10` |
| Empirical null cutoff | `6.91e-11` | `8.00e-11` |

The two studies have independent random streams and their own empirical cutoffs; this is not a paired causal estimate of the exact error's effect size. The exact deterministic difference is the extra `0.5 K_right` term in the old `other`-region generator covariance; the number of near-zero LRT-only decisions varies with null sample and fit noise. Both full rules reject the contrast as non-estimable.

Corrected run-002 rates: uniform material 16/200 adaptive splits (8%); strong contrast 200/200; no right-child paths 0/200; weak contrast/high hit error 0/200; algebraic confounding 0/200. There were no fit failures in the algebraic case, but one coarse-fit nonconvergence in the weak/high-error case (replicate 173; training seed `5030103731`). That failure is retained and is the reason this audit stops before running the new low-budget grid.

## 2. Frozen run-001 ablation

The ablation re-reads the existing [`run-001` trial CSV](studies/adaptive-prototype-001/results/run-001/trial_metrics.csv); it does not regenerate data or refit models. It compares:

- calibrated empirical LRT-only;
- BIC (`LRT > log(n)` for one extra child parameter), a conventional model-selection comparator on the same fitted likelihoods;
- calibrated LRT plus joint-Fisher contrast estimability;
- Fisher estimability plus the declared 0.80 power criterion, without an observed-LRT gate;
- the nested LRT + estimability + power rule and the full convergence-gated rule;
- fixed coarse, fixed fine, and event-count-only references.

| Scenario (200 records each) | Calibrated LRT-only | BIC | Full composite | What the gate changed |
|---|---:|---:|---:|---|
| Uniform material | 6/200 (3%) | 0/200 | 6/200 (3%) | Estimability and power changed no decisions; observed-LRT evidence rejected 194/200 cases relative to the power-only rule. The full rule still had 6 false splits. |
| Strong contrast + independent empty-field calibration | 200/200 | 200/200 | 200/200 | No decision change: all methods with a data-evidence/model-comparison gate detected this large signal. Fine-model held-out NLL was lower than coarse by 46.35 on average. |
| Right child unobserved | 0/200 | 0/200 | 0/200 | Coarse model selection already rejected; Fisher adds an explicit “not estimable” diagnosis, rather than a generic lack of evidence. Fixed fine forced a split. |
| Weak nonzero contrast, 1.0 mm hit error | 11/200 (5.5%) | 2/200 (1%) | 0/200 | The power gate removed 11 LRT-supported but below-minimum-effect splits. This is correct for the declared `|delta|>=0.5` decision, not for detecting every nonzero difference. |
| Exact algebraic nuisance/contrast confounding | 21/200 (10.5%) | 0/200 | 0/200 | Fisher estimability rejected all 21 LRT-only selections. This is the clearest unique operational role of the identifiability gate in run-001. |

The power-only diagnostic selected 200/200 uniform cases because design power for the predeclared `delta=0.5` can be high even when the observed data show no contrast. The LRT gate therefore contributes a separate observed-evidence requirement. Conversely, BIC was more conservative than the full method on the uniform set (0 vs 6 splits) and matched it on the strong signal. The full rule is **not shown to outperform BIC overall** in these conditions. Its demonstrated added capability is explicit contrast estimability handling and enforcement of a declared physically meaningful effect scale; whether those safeguards yield a better trade-off at genuinely low budgets remains unresolved.

All five rows have nested gate changes of 0 for estimability+LRT to +power in uniform and strong cases; 11 weak-case decisions changed; and 21 algebraic-confounding decisions changed at the estimability gate. The separate “power-only” diagnostic demonstrates why power cannot substitute for observed evidence. In run-001 all fits converged, so the full convergence gate changed zero decisions there.

The corrected audit output is [`adaptive-audit-fix-001/results/run-002/`](studies/adaptive-audit-fix-001/results/run-002/). Its `run001_ablation_by_trial.csv` retains each source record alongside all decision flags; `run001_ablation_summary.json` gives rates and held-out predictive NLL for each comparator. A preliminary audit output at `adaptive-audit-fix-001/results/run-001/` is retained but superseded: its per-trial flags were correct, while the first summary pass compared Python booleans against CSV strings and incorrectly counted them as false. A regression test now covers both representations.

## 3. Near-zero LRT under exact non-identifiability

For run-001's algebraic negative control, the coarse and fine covariance families are exactly equivalent after allowing the confounded nuisance parameter. Across 200 records, raw training LRT ranges from 0 to `1.71e-10`, with mean `1.96e-11`. The unchanged empirical cutoff is `6.91e-11`, so LRT-only selects 21/200. The maximum is only `1.52e-14` of the held-out-NLL objective scale (about 68 binary64 epsilon units at that scale). Because run-001 omitted fitted training NLL values, this ratio is an order-of-magnitude comparison using held-out NLL as an explicit proxy; the raw LRT and cutoff are exact stored values.

These positive near-zero numbers are optimizer/floating-point residue between equivalent likelihood models, not resolved statistical evidence. No scientific LRT threshold was changed after observing them. The Fisher null-space check independently reports the child contrast non-estimable, and the composite rule rejects all cases. Corrected run-002 repeats the same qualitative result under a correctly specified generator.

## 4. Low-budget work: predeclared, not executed

The six-cell grid is fixed in [`ADAPTIVE_AUDIT_FIX_PLAN.md`](ADAPTIVE_AUDIT_FIX_PLAN.md): uniform `N=500/1000`; moderate `delta=0.5/0.75` at `N=500`; `delta=0.5` at `N=1000`; and `delta=0.5, sigma_hit=1.0 mm` at `N=500`. Child-path fractions remain 0.4/0.4/0.2; each cell would use 200 development/evaluation replicates and a distinct 200-replicate null calibration per budget/error stratum. The estimator thresholds and minimum effect remain unchanged.

This grid has **not** been run. In corrected run-002, a weak/high-error coarse fit stopped at iteration 8 with preconditioned projected gradient `1.52e-6` against the fixed `1e-6` optimizer tolerance. The next Armijo decrease was about `2e-14`, below objective rounding fluctuations of order `1e-11`; line search exhausted its 50 halvings and returned nonconvergence. No likelihood covariance was singular. Raising tolerance or selectively rerunning that seed would make the stopping rule outcome-dependent, so this branch records the failure and pauses the new grid pending review. `low_budget_summary.json` in audit run-002 explicitly records `not_run`.

## 5. Geant4 source vs fixed-momentum model

`B1/src/PrimaryGeneratorAction.cc` currently defaults to `Ideal` and calls `G4ParticleGun::SetParticleEnergy(4 GeV)`. The official Geant4 application-developer guide describes `SetParticleEnergy` / `/gun/energy` as setting **kinetic energy** ([Geant4 11.4 UI guide](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Control/userInterfaceCommand.html); [primary generation guide](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/GettingStarted/eventDef.html)). For a muon mass of 0.105658 GeV/c²:

- `T=4 GeV` gives `p=sqrt(T(T+2m))=4.1042986 GeV/c`, 2.6075% above 4 GeV/c;
- `p=4 GeV/c` corresponds to `T=sqrt(p²+m²)-m=3.8957368 GeV`;
- under the leading `1/p²` scattering-variance scaling, using 4 GeV/c instead of the current 4 GeV kinetic-energy source implies an approximately 5.02% lower variance scale.

This is a model/source mismatch, not evidence that the simplified forward model fails; both assumptions must be matched in a controlled diagnostic. `SemiRealistic` is also a kinetic-energy source (2–8 GeV uniform, plus an angular draw) and is not represented by a fixed-4-GeV/c forward model.

### Proposed minimum development diagnostic (not run)

1. Keep the `Ideal` direction and QBBC physics list. Compare two source settings: current `T=4 GeV` and exact `p=4 GeV/c` (equivalently `T=3.8957368 GeV`). Cross each with empty target and current material target: four configurations total.
2. First pilot: 1,000 generated events/configuration (4,000 generated total), distinct recorded seeds. This budget is only for source/geometry/hit-identity pipeline and gross residual checks; do not infer a 5% variance shift from it.
3. Put all outputs under `B1/diagnostics/sci-adaptive-momentum-identity-001/`, with an explicit absolute `MUON_TRACK_OUTPUT` for each CSV. Record command, source commit, Geant4 version, macro, random seed pair, target setting, energy variable/type, code/config/input/output SHA-256, and complete/incomplete event counts.
4. Extend only the diagnostic records: per event identify the primary by `ParentID==0 && PDG==-13` (record its observed TrackID rather than assuming a fixed numeric ID); at each of four detector first crossings record PDG, TrackID, ParentID and GlobalTime. Count events whose four selected hits all match that event's initial μ−, and keep identity fields out of reconstruction inputs.
5. Pilot stopping rule: do not automatically top up. Review the fraction of same-primary-μ− complete tracks and residual-covariance precision before selecting a confirmatory budget. Under the ideal two-independent-Gaussian-sample scalar variance approximation, resolving a 5% variance difference at two-sided 5% alpha and 80% power takes roughly 12,600 complete tracks per sample; four configurations imply at least ~50,400 complete tracks before allowance for acceptance, material/empty subtraction and nuisance estimation. This is a planning lower bound, not a frozen final protocol.

## Verification

- Baseline on the audit branch before edits: 43 tests passed.
- After correction and audit tooling: 49 tests passed; the generator/likelihood covariance regression was observed red before implementation and green after.
- run-002 output files and audit run-002 output hashes are stored with their evidence manifests.
- No Geant4 build or run, no historical CSV modification, and no `main` change.
