# Six-cell execution-entry gate: unit-test evidence only

Date: 2026-10-21. Base: `35e6553b24cb5db880944b3938250fc3aef2fad3`.
Report: [LOW_BUDGET_EXECUTION_GATE_REVIEW.md](../../LOW_BUDGET_EXECUTION_GATE_REVIEW.md).
Protocol: [LOW_BUDGET_EXECUTION_GATE_PLAN.md](../../LOW_BUDGET_EXECUTION_GATE_PLAN.md).

Canonical [unit-tests/run-001](unit-tests/run-001/) contains 82 passing unit
tests, 21 deterministic fault-fixture folders, code snapshots, input/count/seed
records, partial calibration CSVs, failure JSONL and lifecycle manifests.
The fixtures are zero arrays/hand-picked parameters with injected solver
states; none is a six-cell experiment or independent scientific validation.

Reproduce only the complete unit suite and archive a fresh evidence folder:

```sh
PYTHONPATH=sci_method/src python3 sci_method/studies/run_execution_gate_unit_tests.py --run-id run-new-unit-tests-001
```

An existing run ID is rejected before writing. No study runner is called by
this command; unittest invokes the calibration loop only with deterministic
faults that stop at its first or second sample, and never draws study RNG data.
Other baseline tests retain their pre-existing small numerical fixtures.

After ChatGPT review approves actual execution, the protected entry is:

```sh
PYTHONPATH=sci_method/src python3 sci_method/studies/run_adaptive_audit_scenarios.py --run-id run-reviewed-six-cells-001
```

This second command was **not run** during this task. It retains all original
six scenarios, 200 calibration repetitions per declared stratum, 200 train/
validation repetitions per cell, seed formulas, effect/alpha/power and metrics.
The stable LRT is diagnostic; original raw statistics still calibrate the
cutoff and drive the existing methods. BIC comparisons are checked too.

On a numerical gate exception, inspect `run_state.json`, `failure.json`,
`failures.jsonl`, `fits.jsonl`, `inputs.jsonl` and `lrt_checks.jsonl`. A partial
CSV is not a completed research result. Do not resume in place, replace seeds,
discard failures or reinterpret incomplete calibration as a valid threshold.

`evidence.json` preserves hashes on completed and normally aborted Python
runs, including `covariance_optimizer.py` and the gate adapter. Hard process
kill/disk failure can leave a startup manifest plus flushed partial journals.
No new runtime dependency was introduced.
