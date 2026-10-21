# Six-cell execution package — for review, not executed

Date: 2026-10-21. See the [frozen six-cell plan](../../ADAPTIVE_AUDIT_FIX_PLAN.md)
and [numerical review](../../OPTIMIZER_STABILITY_REVIEW.md).

The solver-level fixed panel passes numerical checks. Before launching the
existing batch runner, review and implement the following small logging/stop
integration; it is not a new scientific model or a changed threshold:

1. Write detailed status, reason, fixed-scale KKT, parameters and training NLL
   for every coarse/fine fit, including unsuccessful null calibrations.
2. Flush records and stop if any calibration fit is not numerically certified.
   The existing runner collects only converged LRTs for calibration; do not
   silently compute a cutoff from a selected subset of null replicates.
3. Compare raw NLL-subtraction LRT with stable covariance-difference LRT.
   Preserve both. If the frozen cutoff lies within the declared numerical
   allowance, report an unresolved numerical decision and stop for review;
   never tune alpha, cutoff, effect or power to remove it.
4. Include `covariance_optimizer.py` in all source snapshot/hash lists. Save
   raw synthetic arrays or sufficient statistics plus seeds, observed path
   allocation, exact config, environment, command, code SHA and test results.
5. Allocate a fresh result path; preserve every earlier run. Check all relevant
   history hashes before and after. Do not activate automatic follow-on runs.

Fixed configuration (unchanged from the prior plan):

| Cell | N | delta | sigma_hit mm |
|---|---:|---:|---:|
| uniform_n500_sigma03 | 500 | 0 | 0.3 |
| uniform_n1000_sigma03 | 1000 | 0 | 0.3 |
| moderate_d05_n500_sigma03 | 500 | 0.5 | 0.3 |
| moderate_d075_n500_sigma03 | 500 | 0.75 | 0.3 |
| moderate_d05_n1000_sigma03 | 1000 | 0.5 | 0.3 |
| moderate_d05_n500_sigma10 | 500 | 0.5 | 1.0 |

- Known fixed background; path fractions left/right/other = 0.4/0.4/0.2.
- 4 GeV/c; lambda_ref=0.01/mm; minimum meaningful delta=0.5.
- Alpha=0.05; design power gate=0.80; 200 calibration replicates/stratum and
  200 train/validation replicates/cell.
- Null strata remain (500,0.3),(1000,0.3),(500,1.0),(1000,1.0), including the
  previously declared stratum not directly used by a test cell. Do not replace
  it with a new scan.
- Calibration seed base 4026102001; evaluation base 4027102001. Preserve
  stratum/scenario stride 1000000, trial stride 10, validation offset +1,
  group order and generator implementation. Never replace a failed seed.

Draft command after the integration above and ChatGPT review approval:

```sh
PYTHONPATH=sci_method/src python3 -m unittest discover -s sci_method/tests -v
PYTHONPATH=sci_method/src python3 sci_method/studies/run_adaptive_audit_scenarios.py --run-id run-optimizer-reviewed-001
```

The command is not an execution approval; `run-optimizer-reviewed-001` is a
proposed unused ID. If already present, select a new ID without overwriting it.
Do not pass `--ablation-only` for the eventual six-cell execution; that switch
performs old-record reanalysis only. Results remain development evidence.

Acceptance records: numerical fit status counts, all excluded/failed cases,
null false-split counts with binomial intervals, per-cell split/refusal rates,
same-observation coarse/fine/adaptive/LRT/BIC comparisons, held-out predictive
NLL and compute time. Report all cells, including negative results. Do not
relabel these synthetic runs as independent physical validation.
