# Incomplete diagnostic attempt — retained

Source commit: `0327eca` (full commit in git history).
The legacy solver reproduced A/B and the first two controls. The diagnostic
reader then incorrectly expected historical scatter rows only for nonempty
groups. The historical archive also stores a zero-event row for the uncovered
right child. Execution stopped with `historical sufficient-statistic rows missing`.
This is an audit-reader bug, not missing historical data or a solver failure.
All emitted traces are retained; this run is incomplete and has no final
evidence manifest. The reader is corrected before using a fresh run ID.
