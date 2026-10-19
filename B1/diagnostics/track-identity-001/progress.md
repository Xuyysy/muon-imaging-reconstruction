# Work ledger — Geant4 track identity audit

Base: `bad85e876cbb245813814b62aa0db548d615367b` on isolated branch `codex/geant4-track-identity-audit`.

- Ruling: remote freshness could not be checked because GitHub DNS resolution failed; the available local `origin/codex/sci-forward-prototype` ref exactly matches the user-specified reviewed base, and local `origin/main` is an ancestor. Continue from that base; disclose that remote freshness was not verified.
- Ruling: the hidden Superpowers ledger helper cannot write within the managed worktree under current sandbox permissions; use this visible, tracked audit ledger instead and do not retry the helper.
- Pre-flight: diagnostic C++ event records feed the Python summarizer through dedicated CSV only; the summarizer does not write or consume the baseline track CSV.
- Pre-flight: all current-rule and primary-rule coordinates are event-local and use the existing pre-step detector-selection order; all detector-entry rows separately carry processing sequence and Global Time for earliest-time comparison.
- Task 1: complete — 9 summary tests and 2 run-validator tests pass; the original red test correctly failed on missing summarizer, and one later fixture mismatch (short detector name vs. Geant4 logical volume name) was corrected without changing implementation semantics. Clean Geant4 11.2.2 build passed.
- Task 2: complete — standard-library test command `PYTHONPATH=B1/diagnostics/track-identity-001/scripts python3 -m unittest discover -s B1/diagnostics/track-identity-001/tests -v` reports 11/11 passing. Summary separates overlapping flags, excludes unknown identities from the anomaly denominator, compares coordinate sets only when both complete, and reports a two-sided 95% Wilson interval including a nonzero zero-count upper bound.
- Task 3: in progress — two-seed macros, guarded runner, structural validation, anomaly geometry delta output, and provenance hashing are ready. Next run the 1,000-event gate; 10,000-event run must not start unless it passes.
