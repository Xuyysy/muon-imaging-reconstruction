#!/usr/bin/env python3
"""Fail fast when event identity fields or Geant4 accounting are incomplete."""

import argparse
import csv
import json
import re
from pathlib import Path

from summarize_identity import LAYERS


def validate_run(event_rows, summary, run_log, expected_events):
    errors = []
    if "[Muon source] IDEAL mode: fixed 4 GeV" not in run_log:
        errors.append("Geant4 log does not confirm the configured Ideal fixed-4-GeV source")
    lower_log = run_log.lower()
    if "command not found" in lower_log or "parameter out of range" in lower_log:
        errors.append("Geant4 log reports a rejected macro command or parameter")
    event_ids = []
    current_complete = 0
    for row in event_rows:
        try:
            event_ids.append(int(row["event_id"]))
            count = int(row["primary_candidate_count"])
        except (KeyError, TypeError, ValueError):
            errors.append("event row has invalid event_id or primary_candidate_count")
            continue
        if count != 1:
            errors.append(f"event {row['event_id']} has {count} primary mu- candidates; expected exactly one")
        if count == 1:
            for field in ("primary_track_id", "primary_pdg_id", "primary_parent_id", "primary_global_time_ns"):
                if not row.get(field):
                    errors.append(f"event {row['event_id']} has no {field}")
            if row.get("primary_pdg_id") != "13" or row.get("primary_parent_id") != "0":
                errors.append(f"event {row['event_id']} primary candidate is not PDG 13 with Parent ID 0")
        is_complete = str(row.get("current_complete", "0")).lower() in {"1", "true"}
        current_complete += int(is_complete)
        primary_complete = str(row.get("primary_complete", "0")).lower() in {"1", "true"}
        if primary_complete and count != 1:
            errors.append(f"event {row['event_id']} is primary-complete despite ambiguous primary identity")
        if is_complete:
            for layer in LAYERS:
                for suffix in ("track_id", "pdg_id", "parent_id", "global_time_ns", "x_mm", "y_mm", "z_mm"):
                    if not row.get(f"sel_{layer}_{suffix}"):
                        errors.append(f"complete event {row['event_id']} has empty selected {layer} {suffix}")
        if primary_complete:
            for layer in LAYERS:
                for suffix in ("track_id", "pdg_id", "parent_id", "global_time_ns", "x_mm", "y_mm", "z_mm"):
                    if not row.get(f"primary_{layer}_{suffix}"):
                        errors.append(f"primary-complete event {row['event_id']} has empty {layer} {suffix}")

    if len(event_rows) != expected_events:
        errors.append(f"event CSV has {len(event_rows)} rows; expected {expected_events}")
    if sorted(event_ids) != list(range(expected_events)):
        errors.append("event IDs are not exactly the unique run-local sequence 0..N-1")
    if summary.get("generated_events") != expected_events:
        errors.append("summary generated_events does not match the requested run size")
    if summary.get("current_complete_events") != current_complete:
        errors.append("summary and event-row current-complete counts disagree")
    match = re.search(r"Complete tracks:\s*(\d+)", run_log)
    if not match:
        errors.append("Geant4 log has no Complete tracks count")
    elif int(match.group(1)) != current_complete:
        errors.append("Geant4 RunAction and event diagnostic complete counts disagree")
    return errors


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    args = parser.parse_args()
    errors = validate_run(
        _read_csv(args.events),
        json.loads(args.summary.read_text(encoding="utf-8")),
        args.log.read_text(encoding="utf-8"),
        args.expected_events,
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"PASS: {args.expected_events} unique events; primary mu- identity fields and current-complete counts are consistent.")


if __name__ == "__main__":
    main()
