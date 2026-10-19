# Geant4 Track Identity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the preserved four-layer first-hit rule can combine detector hits from particles other than the event's generated primary μ−, using two separately seeded Geant4 runs (1,000 and 10,000 events).

**Architecture:** Keep the existing hit selection, track CSV schema, and PoCA calculation unchanged. Add parallel event-local identity and detector-entry records, plus a separate Python summarizer that derives overlap-aware counts and Wilson 95% binomial intervals without feeding identity fields into reconstruction.

**Tech Stack:** Existing Geant4 C++ actions, CMake, Python standard library `unittest`, CSV/JSON, shell run scripts.

**Spec:** User-approved “Geant4 四层探测器轨迹身份一致性诊断” request in this task.

## Global Constraints

- Use the reviewed base `bad85e876cbb245813814b62aa0db548d615367b` and do not merge into `main`.
- Preserve the original four-layer first-hit selection and PoCA output; identity data goes to separate diagnostics only.
- Identify the event's initial μ− by observed primary-track properties, not a hard-coded Track ID.
- Run only the specified Ideal 4 GeV μ− source, existing four-layer geometry, default centered U target and QBBC; do not run any new algorithm validation.
- Use distinct explicit RNG seeds and separate output directories for 1,000 and 10,000 events; never write the shared `muon_tracks.csv`.
- Report unknown identities, overlapping categories, event denominators, and a binomial interval; zero observed anomalies must retain a nonzero upper confidence bound.

## Review Focus

- Existing first-hit is based on Geant4 processing order, not necessarily the smallest physical Global Time: log every detector boundary entry and compare both.
- Geant4 primary μ− identification must handle its actual PDG encoding and discovered Track ID; test candidate uniqueness and absence explicitly.
- A full current-rule event with incomplete primary hits is not coordinate-comparable: report it separately and compare acceptance sets over all generated events.
- Diagnostic identity fields must never enter the baseline reconstruction CSV or SCI observation reader.
- The two run event IDs are local to their independent runs; retain run identity and seeds in every provenance record.

---

### Task 1: Add independent identity records

**Files:**
- Modify: `B1/include/EventAction.hh`
- Modify: `B1/src/EventAction.cc`
- Modify: `B1/src/SteppingAction.cc`
- Modify: `B1/include/RunAction.hh` only if run-level diagnostic counts are required

**Interfaces:**
- Preserve `EventAction::SetHit(detector, x, y, z)` semantics; extend it internally with current-step identity and time.
- Add event-local methods to observe primary μ− tracks and detector entry steps.
- Emit one event summary row for every generated event and one boundary-entry row for every observed detector entry into files configured independently of `MUON_TRACK_OUTPUT`.

- [x] **Step 1: Write failing unit tests** for summary classification and denominator handling in `B1/diagnostics/track-identity-001/tests/test_summary.py`.
- [x] **Step 2: Run tests and confirm they fail** because the summarizer is not implemented.
- [x] **Step 3: Implement event-local C++ records** for PDG, discovered Track ID, Parent ID, Global Time, processing order, detector, and position; retain the original selected coordinates and PoCA path.
- [x] **Step 4: Verify separate schemas**: baseline track CSV remains byte-schema-compatible, identity data appears only in dedicated diagnostic CSVs.

### Task 2: Implement overlap-aware summary

**Files:**
- Create: `B1/diagnostics/track-identity-001/scripts/summarize_identity.py`
- Test: `B1/diagnostics/track-identity-001/tests/test_summary.py`

**Interfaces:**
- `summarize(event_rows: Iterable[dict], boundary_rows: Iterable[dict]) -> dict`
- Emit per-run counts, mutually exclusive primary classification, overlapping requested indicators, first-processing-vs-earliest-time checks, Wilson 95% intervals, and zero-event upper bounds.

- [x] Test complete, incomplete, mixed-track, secondary-only, primary-missing, unknown-identity, coordinate-difference, and zero-anomaly cases.
- [x] Verify exact event-total accounting and that overlapping indicators are not incorrectly summed.
- [x] Run the full standard-library unit suite.

### Task 3: Reproduction and provenance

**Files:**
- Create: `B1/diagnostics/track-identity-001/run_1000.mac`
- Create: `B1/diagnostics/track-identity-001/run_10000.mac`
- Create: `B1/diagnostics/track-identity-001/run_audit.sh`
- Create: `B1/diagnostics/track-identity-001/README.md`

- [x] Pin separate RNG seed pairs and event counts in each macro; document run independence and the existing default QBBC/target/source configuration.
- [x] Build in an isolated worktree-local build directory and capture Geant4 version, configure/build command, logs, source commit, and hashes.
- [x] Run 1,000 events first; stop to repair if counts or fields are inconsistent.
- [x] Only after 1,000-event checks pass, run 10,000 events to a distinct result directory.
- [x] Summarize outputs, inspect event totals and all required metrics, and write anomalous-event notes without changing historic reconstruction.

### Task 4: Final review and delivery

- [ ] Re-run build, unit tests, 1,000-event checks, and 10,000-event checks against the final sources.
- [ ] Verify no output path points to shared/historical data; verify all hashes and provenance are present.
- [ ] Review the complete diff, commit only the audit work, and push the independent branch if GitHub is reachable.
- [ ] Return commit SHA and scientific caveats; do not merge or proceed to a track-matching method.
