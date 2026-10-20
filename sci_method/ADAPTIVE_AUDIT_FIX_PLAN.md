# Adaptive prototype audit-fix plan

**Base:** `4a179a753fc1db5e5d488c43831e3edfa4afac7e`  
**Branch:** `codex/sci-adaptive-audit-fix`

## Scope

Correct and test algebraic-confounding covariance parity; analyze the existing run-001 decisions without regenerating its observations; run only a few predeclared, small synthetic event-budget/effect-size cases with unique run IDs; clarify finite-precision LRT behavior; document the Geant4 momentum-source mismatch and a no-run implementation protocol. No historical result, existing Geant4 CSV, SCI result or `main` changes; no Geant4 execution.

## Implementation sequence

1. Add a regression comparing generated covariance and the likelihood covariance for left/right/other paths in the algebraic negative control. Factor the algebraic background derivative into one shared function; require the other-region kernel to be an explicit zero matrix. Preserve `run-001`; create a new audit-fix run ID.
2. Use run-001 per-trial LRT, estimability, power, and decision fields to compute nested ablations: LRT-only, estimability + LRT, power + LRT, and full rule. Keep the existing empirical LRT cutoffs and decisions; no re-calibration or threshold changes from outcomes.
3. Quantify exact/near-zero LRT scores under non-identifiability against optimizer tolerances and objective roundoff. Label them numerical artifacts; retain the existing LRT threshold and make estimability a separate guard.
4. Predeclare six cells only: uniform at N=500 and 1,000 (sigma=0.3 mm); delta=0.5 and 0.75 at N=500, sigma=0.3 mm; delta=0.5 at N=1,000, sigma=0.3 mm; delta=0.5 at N=500, sigma=1.0 mm. Keep path fractions 0.4/0.4/0.2, allocate rounded integer event counts, calibrate 200 independent nulls per N/error stratum, and evaluate 200 independent train/validation trials per cell. No broad parameter search. Record false split, missed target split, predictive NLL, and fitting/decision costs.
5. Inspect Geant4 generator and run macro configuration; document kinetic energy vs momentum and the fixed-4-GeV/c model assumption. Specify, but do not run, a small paired empty/material diagnostic with identity, seed, hashes, and isolated output directory.
6. Run unit tests, rerun only declared synthetic cells, verify all output hashes, document evidence boundaries, review diff, commit and push this branch only.

## Geant4 development diagnostic (design only; no run in this branch)

- Four distinct configurations: empty/material target crossed with (a) current `SetParticleEnergy(4 GeV)` and (b) exact `p=4 GeV/c` source. Use fixed Ideal source direction and existing QBBC, but independent recorded seeds per run.
- Initially generate at most 1,000 events per configuration (4,000 total) solely as a pipeline/identity diagnostic. Save every generated event's primary μ− identity plus each detector plane's first-crossing PDG, TrackID, ParentID and GlobalTime; do not feed identity fields to reconstruction. Report all-event and same-primary-μ− four-plane completeness.
- Use `B1/diagnostics/sci-adaptive-momentum-identity-001/` with separate raw CSVs, JSON configs, logs and SHA-256 manifest. Use the existing `MUON_TARGET_SCENARIO=empty|...` and `MUON_TRACK_OUTPUT` controls; never write to the shared baseline CSV.
- The 1,000-event stage is descriptive and not powered to distinguish a roughly 5% scattering-variance scale shift. Under an ideal scalar-Gaussian variance approximation, a two-sided 5% test with 80% power to resolve a 5% variance difference needs about 12,500 independent complete tracks per compared sample; detector acceptance, empty/material subtraction and nuisance estimation may require more. Freeze final physical sample size only after reviewing the pilot and the primary endpoint.

## Frozen decision definitions for ablation

- **LRT-only:** split iff the training LRT exceeds the already calibrated run-001 stratum cutoff.
- **Estimability + LRT:** additionally require the child contrast to be estimable in the joint data Fisher.
- **Power + LRT:** additionally require declared-design Wald power >= 0.80; this diagnostic ablation does not waive the estimability check if power is undefined.
- **Full:** the existing production rule requiring converged fits, estimability, power >= 0.80, and LRT strictly above the existing cutoff.

For an ablation that omits a prerequisite, results are diagnostic and do not define a new production rule. Reuse the same run-001 trial records; do not tune thresholds.
