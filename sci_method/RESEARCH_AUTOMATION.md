# Scientific GitHub automation — governance and usage

Status: **proposed automation on an isolated research branch, NOT merged into main**.
This document records workflow behavior, not a scientific result.

## Primary day-to-day workflow

1. Researcher approves scientific question, empirical protocol, cost ceiling, and any material changes.
2. Choose the tool for **scientific and engineering quality**, not by a rigid GitHub-first hierarchy.
   ChatGPT directly uses the connected GitHub integration for suitable repository operations,
   evidence management, and independent audit. **Proactively recommend Codex whenever it
   is likely to produce materially better mathematics, science, coding, or verification,
   even for a task that could technically be done through GitHub alone.**
   Before any Codex handoff, explain its concrete advantage, recommended model/effort,
   exact scope, starting ref/branch, deliverables, cost ceiling and prohibited operations,
   and **obtain the researcher's explicit approval**. Do not silently invoke Codex or
   assume approval based on previous tasks. Codex may challenge the plan within the
   separately approved scope and budget.
3. All changes live on an isolated branch. GitHub Actions verifies ordinary code and
   immutable evidence, collects logs and a CI report; it does **not** infer scientific novelty.
4. ChatGPT examines code, mathematical assumptions, statistical validity, negative
   results and changes versus independent strong baselines, then reports pass/blockers.
5. Researcher decides whether to approve the next experiment, training, Geant4,
   new method, mainline merge or manuscript claim.
6. After independently reviewing each completed research stage, update the paper-note
   evidence index and the two Notion scientific records, including cross-chat handoff.
   GitHub remains authoritative for code, seeds, configs and raw results.

## This first installation

- **`.github/workflows/sci-evidence-ci.yml`**: automatic on PRs touching research/CI
  files and on pushes to this setup branch; also declares manual dispatch.
  Jobs run on Linux, one-thread BLAS, pinned Python/NumPy and immutable Action SHAs.
  CI checks archived SHA-256 provenance and old-result overwrite protection, then
  runs **only** the 20 bounded `ml-nonneural-feasibility-001` tests.
  CI uploads a report/logs as short-retention Actions artifacts (14 days).
- **`.github/workflows/sci-archive-replay.yml`**: manual-only archival integrity
  and replay of **existing** stage-1 RNG keys and 6,144 selection/audit records.
  Read-only input; no new experiment or run-ID output. Limited to 8 min and a
  180-second replay step. The manual trigger becomes available only when the
  workflow exists on the repository default branch; in this isolated branch it is
  **prepared, not enabled for direct dispatch**.
- **`.github/scripts/sci_evidence_gate.py`**: source-visible, read-only validation.
  A PR altering an *existing* study `results/` or `verification/` file fails
  protection rather than quietly accepting changed history. New runs with distinct
  paths can be added only under separately approved research protocols.

GitHub Actions checks are **not** an independent scientific replication or an SCI
publication claim. Never turn an archived diagnostic into a confirmatory study just
because CI passed. CPU-time and hosted-runner minutes are not equivalent to the
previously granted research experiment budget; no uncontrolled compute is authorized.
Workflow costs, GitHub billing, and repository Actions permissions need review before
expanding jobs, schedules, or data uploads.

## Deliberate safety boundaries

- No GitHub secret access, no personal-data uploads, no write permissions in workflows.
- No automatic branch merge, no direct changes to `main`, no force push, no rewriting
  previous runs, no loss of negative results, no moving frozen thresholds.
- No automatic Geant4, CNN/GPU, new Monte Carlo research, unbounded exploration,
  scheduled research experiments, or live scientific claim promotion.
- Predeclared and exploratory checks must retain distinct labels. A failing archived
  cross-platform RNG replay is a provenance issue for investigation; do not silently
  regenerate seeds, edit observations, or mark it as a scientific failure.

## Review and activation

The isolated setup branch is based on the independently reviewed B2 stage-1 research
commit `7626414945133244d39842abd45cfc7033f700be`.
The first draft PR should target `codex/sci-ml-nonneural-feasibility` to avoid
accidentally proposing all prior branch-specific research content into the old
`main` history. Do **not** auto-merge it. A separate explicit researcher's decision
is required to integrate reviewed code and workflows into `main`.

Until merged into the default branch, the manual dispatch workflow is defined but
cannot be launched from the Actions UI. After activation, a user still needs to
manually request each replay; the automatic CI does not execute new science.

## Future safe expansion (each step requires scope review)

- Check a wider, explicitly chosen low-budget test matrix and pin environments.
- Emit a structured PR evidence index, failure summaries, negative controls and
  exact artifact hashes; do not edit the paper notebook without scientific review.
- Create an issue template for hypotheses, falsification, baselines, budget and
  source commit; use draft PRs for proposed experiments.
- Add manual-only bounded mathematical diagnostics after the user's specific
  approval. Hold GPU/Geant4/production scanning for separate approval.

Current scientific decision remains **B: no ML training**. The stage-1 mathematical
limits, evidence, and baseline limitations are in
`sci_method/studies/ml-nonneural-feasibility-001/`.
