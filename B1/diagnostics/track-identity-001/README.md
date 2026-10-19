# Geant4 four-layer track identity audit

This directory contains an approved, minimal data-reliability audit of the existing four-layer first-hit rule. It does not change the current reconstruction rule or the SCI forward model, and is not an independent validation of an SCI algorithm.

The Geant4 project currently uses the QBBC physics list, a fixed 4 GeV ideal μ− source, the existing four detector planes, and the default centered U-shaped aluminium target. The source code selects `mu-`; its Geant4/PDG encoding is `+13`. Primary identity is detected dynamically per event using both `ParentID == 0` and PDG ID `13`; the actual Track ID is recorded, never assumed.

## Output separation

- `current_rule_tracks.csv` is the existing first-hit/PoCA schema and is written only to this new diagnostic run directory.
- `event_identity.csv` records the current-rule selection and the initial primary μ− first boundary entry independently for each detector layer.
- `boundary_entries.csv` records every observed transition into any of the four detector volumes with event-local processing order, Global Time, PDG ID, Track ID, Parent ID, and position.
- `identity_summary.json` contains requested event counts, overlapping identity indicators, coordinate/acceptance comparisons, processing-order versus physical-time checks, and a two-sided 95% Wilson interval.
- `anomaly_cases.csv` contains only known identity-anomalous complete events and their descriptive angle/PoCA changes. The run's 99th-percentile angle is a descriptive reference only, not an acceptance threshold.
- `geant4_run.log`, `validation.log`, and final `configuration.json` preserve run details and validation.

The output script refuses to overwrite either run directory. The two Geant4 processes have distinct seed pairs and local event IDs: run-1000 uses `(318527, 742913)` and run-10000 uses `(915731, 460297)`. Equal event IDs across these runs do not identify the same event.

## Build and run

On the machine with the installed Geant4 11.2.2 package:

```sh
cmake -S B1 -B B1/build-track-identity-audit \
  -DGeant4_DIR=/Users/xuyaoyang/geant4-install/lib/cmake/Geant4 \
  -DWITH_GEANT4_UIVIS=OFF
cmake --build B1/build-track-identity-audit --parallel 4
bash B1/diagnostics/track-identity-001/run_audit.sh
```

The runner executes and validates 1,000 events first; it will not proceed to 10,000 if event identity fields, event totals, or the RunAction complete-track count disagree. It sets both `MUON_TRACK_OUTPUT` and the two diagnostic output variables to fresh run-specific paths. It does not read or overwrite `muon_analysis/muon_tracks.csv`.

Run the unit tests with:

```sh
PYTHONPATH=B1/diagnostics/track-identity-001/scripts \
  python3 -m unittest discover -s B1/diagnostics/track-identity-001/tests -v
```

## Classification semantics

The requested categories overlap. “Track IDs inconsistent” means the four selected hits have more than one distinct Track ID. “Selected secondary” means at least one selected hit has `ParentID > 0`. A separate identity anomaly is counted when a current-rule-complete event has a uniquely identified primary μ− and known selected identities, but not all four selected hits match that primary μ− (including all-four-from-one-secondary cases). Unknown primary identity is reported separately and is not silently included in that anomaly denominator. Coordinate comparison uses only events complete under both rules; acceptance differences compare the two completion flags over every generated event.

The Wilson interval denominator is the number of current-rule-complete events with a unique primary μ− and known selected identities. Therefore, a zero anomaly count still has a positive upper confidence bound. Categories are not added together.

Any angle and PoCA changes in `anomaly_cases.csv` are diagnostic recomputations only. They do not replace or modify historical POCA data. The within-run 99th percentile merely describes the tail and is not used as an event-selection threshold.
