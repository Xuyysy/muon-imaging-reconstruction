# Six-cell execution-entry quality gate

Base: `35e6553b24cb5db880944b3938250fc3aef2fad3`.
Branch: `codex/sci-low-budget-execution-gate`. Date: 2026-10-21.
This task runs unit tests only, not the six-cell study or Geant4.

Minimal implementation:

1. Add a study-local adapter to the detailed covariance optimizer. Persist
   initial values, returned status/reason/parameters/NLL/gradient/KKT, and an
   independently recomputed fixed-scale KKT, feasibility, objective consistency
   and interior-curvature check. Keep the solver and statistical rules unchanged.
2. Keep JSONL journals durable with flush+fsync. Save every sample's observed
   count, scatter matrix Y.T@Y and canonical observation-content hash before
   fitting. Seed/phase/replicate and sample roles accompany each record.
3. On any uncertified fit, persist fit/failure records and stop immediately;
   explicitly label any unattempted fine model. Never replace seeds or calibrate
   from a subset. Successful calibration must contain every declared replicate.
4. For known_fixed only, verify coarse-to-fine map
   (parent,other)->(parent,parent,other), identical observations/base covariance,
   and derivative identity Dc=Df@mapping. Preserve the old raw signed NLL
   subtraction and max(0,raw) statistic. Compute a diagnostic stable LRT using
   the fine-model covariance difference; do not replace the scientific cutoff.
5. Use the prior numerical audit budget per fit
   b=max(1e-8,512*eps*max(1,abs(NLL))) plus observed objective-evaluation
   discrepancies. Pair allowance=2*(bc+bf). Record nonfinite, substantially
   negative, or inconsistent LRTs and abort. The allowance is a conservative
   numerical diagnostic, not a rigorous error bound or statistical confidence
   interval. No alpha/effect/power adjustments.
6. Calibrate on all original raw statistics. Also record the stable-statistic
   cutoff diagnostically with the same order statistic/alpha. Calibration
   allowance=max pair allowance bounds order-statistic perturbations. At
   evaluation, add calibration allowance to the current pair allowance for
   the calibrated cutoff; BIC uses the pair allowance only. If either raw/stable
   comparison disagrees or its interval touches the cutoff, persist and abort.
7. Save config, environment, source commit/hashes/snapshots (including optimizer
   and this adapter) before work. A finally block writes aborted/completed
   state, exception/traceback and an output-hash manifest on every normal Python
   exit path. Preserve partial records on failure. An existing directory is
   rejected before any writes. Hard process kill/power loss remains outside
   the guarantees of a Python finally block; journals are durable per record.
8. Tests inject solver faults into deterministic fixtures, not frozen study
   observations. Test early abort, false convergence, malformed states,
   near-cutoff ambiguity, mapping identity, normal fits and failure manifests.
   Run the entire unit suite; preserve its log and the unchanged frozen plan
   hash. No six-cell command is invoked this task.
