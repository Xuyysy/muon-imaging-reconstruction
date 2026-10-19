#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B1_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$B1_DIR/.." && pwd)"
BUILD_DIR="${MUON_TRACK_AUDIT_BUILD_DIR:-$B1_DIR/build-track-identity-audit}"
RESULT_SET="${MUON_TRACK_AUDIT_RESULT_SET:-audit-001}"
RESULTS_DIR="$SCRIPT_DIR/results/$RESULT_SET"
PYTHON="${PYTHON:-python3}"

if [[ -e "$RESULTS_DIR" ]]; then
  echo "Refusing to overwrite existing audit outputs in $RESULTS_DIR" >&2
  exit 2
fi

if [[ ! -x "$BUILD_DIR/exampleB1" ]]; then
  echo "Missing $BUILD_DIR/exampleB1; configure and build first (see README.md)." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/run-1000" "$RESULTS_DIR/run-10000"
PYTHONPATH="$SCRIPT_DIR/scripts" "$PYTHON" -m unittest discover \
  -s "$SCRIPT_DIR/tests" -v >"$RESULTS_DIR/unit_tests.log" 2>&1
cmake --build "$BUILD_DIR" --parallel 4 >"$RESULTS_DIR/build.log" 2>&1

run_one() {
  local run_id="$1"
  local macro="$2"
  local expected_events="$3"
  local run_dir="$RESULTS_DIR/$run_id"
  local log="$run_dir/geant4_run.log"

  (
    cd "$BUILD_DIR"
    env \
      MUON_TARGET_SCENARIO=centered \
      MUON_TRACK_OUTPUT="$run_dir/current_rule_tracks.csv" \
      MUON_TRACK_DIAGNOSTICS_OUTPUT="$run_dir/event_identity.csv" \
      MUON_TRACK_BOUNDARY_OUTPUT="$run_dir/boundary_entries.csv" \
      ./exampleB1 "$SCRIPT_DIR/$macro"
  ) >"$log" 2>&1

  "$PYTHON" "$SCRIPT_DIR/scripts/summarize_identity.py" \
    --events "$run_dir/event_identity.csv" \
    --boundaries "$run_dir/boundary_entries.csv" \
    --output "$run_dir/identity_summary.json" \
    --anomalies-output "$run_dir/anomaly_cases.csv" \
    --run-id "$run_id"

  "$PYTHON" "$SCRIPT_DIR/scripts/validate_run.py" \
    --events "$run_dir/event_identity.csv" \
    --summary "$run_dir/identity_summary.json" \
    --log "$log" \
    --expected-events "$expected_events" \
    >"$run_dir/validation.log"
}

echo "Stage 1/2: running 1,000 events with seed pair 318527,742913"
run_one run-1000 run_1000.mac 1000
echo "Stage 1/2 validation passed; proceeding to independent 10,000-event run."

echo "Stage 2/2: running 10,000 events with seed pair 915731,460297"
run_one run-10000 run_10000.mac 10000
echo "Both diagnostic runs completed and passed structural validation."
echo "Results: $RESULTS_DIR"

"$PYTHON" "$SCRIPT_DIR/scripts/write_manifest.py" \
  --repo "$REPO_ROOT" --script-dir "$SCRIPT_DIR" \
  --build-dir "$BUILD_DIR" --results-dir "$RESULTS_DIR"
