# Track identity audit results

## Scope and provenance

These are two separately launched Geant4 11.2.2 serial runs using the existing QBBC physics list, Ideal fixed 4 GeV μ− source, four detector planes, and centered U-shaped aluminium target. The event IDs restart in each process and are run-local. This is a data-reliability audit only; it is not an independent SCI algorithm validation. The source commit used to generate the Geant4 data is recorded in `results/audit-001/geant4_run_manifest.json`; analysis source and file hashes are in `configuration.json`.

The primary μ− was recognized per event as a Geant4 primary track with PDG ID `13` (μ−) and `ParentID == 0`; its actual Track ID was read from the track. The current first-hit rule remained unchanged. The baseline-shaped track CSV and identity diagnostics were written to distinct new paths.

## Counts

| Metric | 1,000 events (318527, 742913) | 10,000 events (915731, 460297) |
|---|---:|---:|
| Generated events | 1,000 | 10,000 |
| Current-rule four-layer complete | 997 | 9,966 |
| Initial μ− four-layer complete | 996 | 9,966 |
| All four current-rule hits from that initial μ− | 996 | 9,966 |
| Current-rule four-hit Track IDs inconsistent | 1 | 0 |
| At least one selected secondary hit (`ParentID > 0`) | 1 | 0 |
| Current vs. primary coordinates differ, among events complete under both | 0 / 996 | 0 / 9,966 |
| Current vs. primary four-layer acceptance differs | 1 | 0 |
| Primary identity uniquely identified | 1,000 | 10,000 |
| Primary identity missing | 0 | 0 |
| Primary identity ambiguous (multiple μ− candidates or invalid unique ID) | 0 | 0 |
| Primary identity unknown (missing + ambiguous) | 0 | 0 |
| Selected-hit identity unknown among current-complete events | 0 | 0 |
| Events with at least one layer whose first processed boundary is not earliest by Global Time | 1 | 4 |
| Event-layer cases with processing/time ordering difference | 1 | 11 |
| Events where current selected hit is later than earliest boundary entry | 1 | 4 |

The identity-anomaly rate is defined only among current-rule-complete events with one uniquely identified primary μ− and known identities for all four selected hits. The categories “Track IDs inconsistent”, “selected secondary”, “coordinates differ”, and “identity anomaly” overlap; they must not be summed.

- 1,000-event run: `1 / 997 = 0.1003%`; two-sided 95% Wilson interval **0.0177%–0.5659%**.
- 10,000-event run: `0 / 9,966 = 0% observed`; two-sided 95% Wilson interval **0%–0.0385%**. The positive upper bound is retained; zero observations do not establish impossibility.
- Optional pooled estimate across the two distinct-seed runs: `1 / 10,963 = 0.00912%`; two-sided 95% Wilson interval **0.00161%–0.05165%**. This pooling describes the same fixed configuration only and does not make the runs an SCI method validation.

For the coordinate comparison, only events complete under both rules are comparable. The one acceptance-difference event is not in that coordinate denominator because the initial μ− did not produce all four detector hits.

## Identity anomaly: run-1000 event 40

The current first-hit rule selected:

- UpstreamDetector1 and UpstreamDetector2: Track ID 1, PDG 13, Parent ID 0 (the initial μ−).
- DownstreamDetector1 and DownstreamDetector2: Track ID 2, PDG 11, Parent ID 1 (an electron secondary from the initial μ−).

Thus, one generated event was accepted as a four-plane track even though its selected detector hits combine the initial μ− with its electron secondary. The primary μ− had only two of the four selected planes, so the primary-only four-point scattering angle and PoCA are undefined for this event; no artificial delta is reported. This changes the current-rule acceptance by one event.

The unmodified C++ current-rule track output gives `theta_x = 134.382 mrad`, `theta_y = -160.129 mrad`, and `theta = 209.045 mrad`. It exceeds the run's descriptive current-angle 99th percentile (9.097 mrad); this percentile is a description, not a selection threshold. Its PoCA midpoint is `(-98.6139, 149.006, -174.362) mm`, well outside the target volume. This is a concrete example of a secondary-hit combination producing a severe, physically implausible POCA observation. The reported current-rule fit is taken from the existing C++ output, not recomputed from the rounded diagnostic coordinates. The historical reconstruction data were not changed.

## Processing order versus physical Global Time

The per-layer case file `boundary_order_cases.csv` preserves both particle identities, times, and positions. In each of the 11 event-layer cases in the 10,000-event run, the current selection/first processed entry was the initial μ− (Track ID 1), while an entry processed later had an earlier Global Time and belonged to a secondary particle (electron or photon). The time lead ranged from approximately `4.3e-5` to `5.9e-4 ns`; first-processed versus earliest-time positions differed by about `1.4–27.0 mm` in the recorded plane. These were four events, not eleven independent events.

This does not mean the existing selection mixed particles in those 10,000-event cases: their four selected hits still matched the initial μ−. It does show that Geant4 track-processing order is not equivalent to earliest physical boundary-crossing time. Earliest boundary crossing is not itself a detector electronics response model, so no alternate hit-selection algorithm was tested here.

## Interpretation and recommended handoff

The current rule can combine particles within an event; it did so once in the 1,000-event run, and the resulting fit was pathological. The 10,000-event run did not reproduce the identity anomaly, but its confidence bound and the observed 1,000-event case mean the possibility cannot be dismissed. This matters to POCA-derived observations and to any SCI forward-model input built from unfiltered four-layer hits: one event can represent an electron's downstream path rather than the primary μ− track.

Before using freshly generated Ideal single-μ− events for SCI reconstruction, review whether the data-production contract should require all four hits to share the event's initial μ− identity (or mark incomplete primary tracks as rejected). This audit does not implement that rule, retrofit old CSVs, build detector-level hit matching, or revise prior POCA results. Such a production change should be decided after reviewing this evidence; historical files remain untouched.
