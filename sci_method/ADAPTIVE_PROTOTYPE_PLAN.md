# Identifiability-calibrated split/merge prototype plan

> **Execution:** Inline in the isolated `codex/sci-adaptive-prototype` worktree, with test-first changes and one final independent code review.

**Goal:** Build a minimal two-child split/merge prototype whose split decision requires an estimable local contrast, sufficient finite-budget information, and calibrated evidence, then evaluate it on independent synthetic events.

**Architecture:** Keep the existing Gaussian covariance forward model and add causal residual response, estimability-aware contrast uncertainty, a small two-child model with separately represented other-region/background terms, and a deterministic synthetic study driver. The decision function receives paths, observations, calibration data, and declared thresholds only; truth labels are used only by the evaluation/reporting layer.

**Tech Stack:** Python standard library and existing NumPy only; unittest; existing Geant4-independent four-plane covariance model at fixed 4 GeV/c.

**Scientific authority:** User request dated 2026-10-20; base `bad85e876cbb245813814b62aa0db548d615367b`; canonical prior synthetic calibration evidence is run-007.

## Global constraints

- Do not run Geant4 or use existing Geant4/POCA results as validation data.
- Preserve all existing data and runs; create only `sci_method/studies/adaptive-prototype-001/results/run-001/`.
- Use 4 GeV/c and detector z `[-400,-300,300,400] mm`. Run-007 remains the canonical prior result source; this new contrast-calibration study uses a separate declared reference scale `lambda_ref=1e-2 mm^-1` and 4,000 events because run-007's `1e-4 mm^-1` / 200-event setting gave only chance-level design power for the predeclared 0.5-reference-unit contrast. This is a new labeled synthetic design, not a rewrite of run-007.
- Keep prior precision separate from data Fisher information.
- Reject an unestimable contrast; never interpret a pseudoinverse finite value as finite estimability when the contrast has a null-space component.
- Use one common observation set and the same Gaussian likelihood for fixed-coarse, fixed-fine, count-only, and information/evidence-driven candidates.
- Calibration and final synthetic evaluation use disjoint, recorded random seeds. Results are prototype/development evidence, not independent SCI validation.
- Preserve the exact algebraic nuisance degeneracy as a labeled negative control, not a physical-background model.

## Review focus

1. A scatter kick downstream of a measured plane must have zero covariance contribution to that plane; scatter between the detector groups must retain the prior valid response.
2. Contrast estimability must be checked against the Fisher null space before using a pseudoinverse variance.
3. Empty-field calibration information must be independent of target observations and add only to the joint data likelihood information.
4. Truth fields must not enter model fitting, split decisions, or prediction scores.
5. Calibration and evaluation seeds, outputs, code snapshot, and hashes must be disjoint/traceable and an existing run path must never be overwritten.

## Tasks

### Task 1 — Causal path-integral response

**Files:** modify `sci_method/src/sci_method/forward_model.py`; tests in `sci_method/tests/test_forward_model.py`.

- [x] Add a failing regression where a voxel wholly downstream of the first downstream detector plane contributes zero to that plane's residual covariance, while it can contribute to the later plane.
- [x] Preserve/add a regression that a voxel between detector groups contributes the existing nonzero downstream covariance.
- [x] Change the response integrand to the causal positive-part lever arms, then run forward-model tests.

### Task 2 — Contrast estimability and uncertainty

**Files:** modify `sci_method/src/sci_method/identifiability.py`; add `sci_method/tests/test_split_merge.py`.

- [x] Add tests for estimable contrast variance/information, exact null-space contrast rejection even when the pseudoinverse quadratic form is finite, and nuisance profiling through the full joint Fisher.
- [x] Implement a scale-aware null-space projection check and return explicit estimability, variance, information, and null projection diagnostics.
- [x] Verify prior precision is not included in the data contrast information.

### Task 3 — Two-child model and split rule

**Files:** add `sci_method/src/sci_method/split_merge.py`; extend `sci_method/tests/test_split_merge.py`.

- [x] Define dimensionless parameters relative to `lambda_ref=1e-2 mm^-1`: parent mean, signed child contrast, other-region nuisance, and optional background nuisance.
- [x] Encode known-fixed background, unknown background with independent empty-field likelihood, and an explicitly algebraic contrast-confounded negative control.
- [x] Implement split eligibility from contrast estimability and design power for a declared physical contrast, plus a likelihood-ratio evidence threshold calibrated only on development null replicas.
- [x] Add the count-only comparator and fixed coarse/fine prediction evaluators; reject truth/mask/oracle keys in decision inputs.

### Task 4 — Reproducible synthetic calibration/evaluation

**Files:** add `sci_method/studies/run_adaptive_split_merge.py`; write exclusively under `sci_method/studies/adaptive-prototype-001/results/run-001/`.

- [x] Generate calibration and independent evaluation data for uniform, strong contrast, one-child-uncovered, weak contrast/high-hit-error, and algebraic-confounding cases, preserving sufficient statistics and per-replicate predictions.
- [x] Compare fixed coarse, fixed fine, event-count split, and calibrated information/evidence split on the same generated train/evaluation events and likelihood.
- [ ] Report correct-split rate, false-split rate, rejection of unidentifiable cases, held-out predictive NLL difference, costs, uncertainty intervals, and all failures.
- [x] Save config, seed schedule, source snapshot, environment, commit/status, input/output SHA-256, and reproduction command.

### Task 5 — Research record and final verification

**Files:** update `sci_method/FORWARD_MODEL.md`, `sci_method/SCI_PAPER_OUTLINE.md`, `sci_method/RESEARCH_EVIDENCE_INDEX.md`, add study README/results.

- [x] Describe the three background cases, contrast estimability/power rule, causal range, scope, and failure cases without claiming SCI validation.
- [x] Point canonical previous numerical values to run-007 and remove stale run-003 citation in the outline.
- [ ] Run the full unit suite, study structural checks, hash verification, source review, and independent code review; do not run Geant4.
- [ ] Commit and push the independent branch only; do not merge.
