#!/usr/bin/env python3
"""Summarize the separate Geant4 track-identity audit CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LAYERS = ("up1", "up2", "down1", "down2")
DETECTOR_NAMES = {
    "UpstreamDetector1": "up1",
    "UpstreamDetector2": "up2",
    "DownstreamDetector1": "down1",
    "DownstreamDetector2": "down2",
}
_WILSON_Z_95 = 1.959963984540054
_POSITION_TOLERANCE_MM = 1.0e-9


def _integer(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _wilson_interval(successes: int, trials: int) -> dict:
    if trials <= 0:
        return {"lower": None, "upper": None}
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    p = successes / trials
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = (_WILSON_Z_95 * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    ) / denom)
    return {"lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def _identity_values(row: dict, prefix: str, layer: str):
    return (
        _integer(row.get(f"{prefix}_{layer}_track_id")),
        _integer(row.get(f"{prefix}_{layer}_pdg_id")),
        _integer(row.get(f"{prefix}_{layer}_parent_id")),
    )


def _coordinates(row: dict, prefix: str, layer: str):
    values = tuple(_number(row.get(f"{prefix}_{layer}_{axis}_mm"))
                   for axis in ("x", "y", "z"))
    return values if all(value is not None for value in values) else None


def _unit(vector):
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        raise ValueError("Cannot fit a track from coincident detector coordinates.")
    return tuple(component / magnitude for component in vector)


def _geometry(row: dict, prefix: str):
    points = [_coordinates(row, prefix, layer) for layer in LAYERS]
    if any(point is None for point in points):
        return None
    up1, up2, down1, down2 = points
    incoming = _unit(tuple(b - a for a, b in zip(up1, up2)))
    outgoing = _unit(tuple(b - a for a, b in zip(down1, down2)))
    theta_x = math.atan2(outgoing[0], outgoing[2]) - math.atan2(incoming[0], incoming[2])
    theta_y = math.atan2(outgoing[1], outgoing[2]) - math.atan2(incoming[1], incoming[2])

    w0 = tuple(a - b for a, b in zip(up1, down1))
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    aa, bb, cc = dot(incoming, incoming), dot(incoming, outgoing), dot(outgoing, outgoing)
    dd, ee = dot(incoming, w0), dot(outgoing, w0)
    denominator = aa * cc - bb * bb
    s = t = 0.0
    if abs(denominator) > 1.0e-12:
        s = (bb * ee - cc * dd) / denominator
        t = (aa * ee - bb * dd) / denominator
    poca_in = tuple(point + s * direction for point, direction in zip(up1, incoming))
    poca_out = tuple(point + t * direction for point, direction in zip(down1, outgoing))
    poca = tuple(0.5 * (a + b) for a, b in zip(poca_in, poca_out))
    return {
        "theta_x_mrad": theta_x * 1000.0,
        "theta_y_mrad": theta_y * 1000.0,
        "theta_mrad": math.hypot(theta_x, theta_y) * 1000.0,
        "poca_x_mm": poca[0],
        "poca_y_mm": poca[1],
        "poca_z_mm": poca[2],
    }


def _event_flags(row: dict):
    current_complete = _flag(row.get("current_complete"))
    primary_complete = _flag(row.get("primary_complete"))
    candidate_count = _integer(row.get("primary_candidate_count"))
    primary_track_id = _integer(row.get("primary_track_id"))
    primary_known = candidate_count == 1 and primary_track_id is not None

    selected_identity = [_identity_values(row, "sel", layer) for layer in LAYERS]
    selected_identity_known = all(
        track_id is not None and pdg is not None and parent is not None
        for track_id, pdg, parent in selected_identity
    )
    selected_track_ids = [identity[0] for identity in selected_identity]
    track_id_inconsistent = (
        current_complete
        and all(track_id is not None for track_id in selected_track_ids)
        and len(set(selected_track_ids)) > 1
    )
    selected_secondary = (
        current_complete
        and any(parent is not None and parent > 0
                for _, _, parent in selected_identity)
    )
    all_initial_muon = (
        current_complete and primary_known and selected_identity_known
        and all(track_id == primary_track_id and pdg == 13 and parent == 0
                for track_id, pdg, parent in selected_identity)
    )
    selected_unknown = current_complete and (
        not primary_known or not selected_identity_known
    )
    anomaly_eligible = current_complete and primary_known and selected_identity_known

    coordinate_comparable = current_complete and primary_complete and primary_known
    coordinates_differ = False
    if coordinate_comparable:
        for layer in LAYERS:
            selected = _coordinates(row, "sel", layer)
            primary = _coordinates(row, "primary", layer)
            if selected is None or primary is None:
                coordinate_comparable = False
                break
            if any(abs(a - b) > _POSITION_TOLERANCE_MM
                   for a, b in zip(selected, primary)):
                coordinates_differ = True

    return {
        "current_complete": current_complete,
        "primary_complete": primary_complete,
        "primary_known": primary_known,
        "primary_unknown": not primary_known,
        "selected_unknown": selected_unknown,
        "track_id_inconsistent": track_id_inconsistent,
        "selected_secondary": selected_secondary,
        "all_initial_muon": all_initial_muon,
        "anomaly_eligible": anomaly_eligible,
        "identity_anomaly": anomaly_eligible and not all_initial_muon,
        "coordinate_comparable": coordinate_comparable,
        "coordinates_differ": coordinate_comparable and coordinates_differ,
        "acceptance_changed": current_complete != primary_complete,
    }


def summarize(event_rows: Iterable[dict], boundary_rows: Iterable[dict]) -> dict:
    events = list(event_rows)
    boundaries = list(boundary_rows)
    flags = [_event_flags(row) for row in events]

    boundary_by_event_layer = defaultdict(list)
    for row in boundaries:
        layer = DETECTOR_NAMES.get(str(row.get("detector", "")))
        event_id = _integer(row.get("event_id"))
        processing_order = _integer(row.get("processing_order"))
        global_time = _number(row.get("global_time_ns"))
        if layer is None or event_id is None or processing_order is None or global_time is None:
            continue
        boundary_by_event_layer[(event_id, layer)].append((processing_order, global_time))

    order_difference_events = set()
    order_difference_layers = 0
    selected_time_difference_events = set()
    event_rows_by_id = {_integer(row.get("event_id")): row for row in events}
    for (event_id, layer), entries in boundary_by_event_layer.items():
        first_processed = min(entries, key=lambda item: item[0])
        earliest_time = min(time for _, time in entries)
        if first_processed[1] > earliest_time + 1.0e-12:
            order_difference_events.add(event_id)
            order_difference_layers += 1
        event_row = event_rows_by_id.get(event_id)
        selected_time = _number(event_row.get(f"sel_{layer}_global_time_ns")) if event_row else None
        if selected_time is not None and selected_time > earliest_time + 1.0e-12:
            selected_time_difference_events.add(event_id)

    generated = len(events)
    complete = sum(flag["current_complete"] for flag in flags)
    eligible = sum(flag["anomaly_eligible"] for flag in flags)
    anomalies = sum(flag["identity_anomaly"] for flag in flags)
    wilson = _wilson_interval(anomalies, eligible)
    return {
        "generated_events": generated,
        "current_complete_events": complete,
        "primary_complete_events": sum(flag["primary_complete"] for flag in flags),
        "all_four_current_from_primary_muon_events": sum(
            flag["all_initial_muon"] for flag in flags
        ),
        "track_id_inconsistent_events": sum(
            flag["track_id_inconsistent"] for flag in flags
        ),
        "selected_secondary_events": sum(
            flag["selected_secondary"] for flag in flags
        ),
        "current_vs_primary_coordinates_differ_events": sum(
            flag["coordinates_differ"] for flag in flags
        ),
        "coordinate_comparable_events": sum(
            flag["coordinate_comparable"] for flag in flags
        ),
        "primary_identity_unknown_events": sum(flag["primary_unknown"] for flag in flags),
        "selected_identity_unknown_events": sum(flag["selected_unknown"] for flag in flags),
        "acceptance_set_difference_events": sum(
            flag["acceptance_changed"] for flag in flags
        ),
        "events_with_processing_vs_earliest_time_difference": len(order_difference_events),
        "layers_with_processing_vs_earliest_time_difference": order_difference_layers,
        "events_with_selected_hit_later_than_earliest_boundary_entry": len(selected_time_difference_events),
        "identity_anomaly": {
            "definition": "current-complete events with uniquely identified primary muon and known selected identities, where not all four selected hits belong to that primary mu-",
            "count": anomalies,
            "denominator": eligible,
            "fraction": anomalies / eligible if eligible else None,
            "wilson_95_interval": wilson,
            "interval_method": "two-sided Wilson score interval, z=1.959963984540054",
            "zero_count_upper_bound": wilson["upper"] if anomalies == 0 else None,
        },
        "overlap_note": "Track-ID inconsistency, secondary-hit, coordinate-difference, and primary-identity anomaly indicators overlap and must not be summed.",
    }


def anomaly_cases(event_rows: Iterable[dict]) -> tuple[list[dict], float | None]:
    """Return descriptive geometry deltas for known identity-anomalous events.

    The 99th percentile is only a within-run descriptive reference; it is not
    an acceptance cut and is never used to select or reject reconstruction data.
    """
    events = list(event_rows)
    current_angles = []
    for row in events:
        if _flag(row.get("current_complete")):
            geometry = _geometry(row, "sel")
            if geometry is not None:
                current_angles.append(geometry["theta_mrad"])
    current_angles.sort()
    p99 = current_angles[math.ceil(0.99 * len(current_angles)) - 1] if current_angles else None

    rows = []
    for row in events:
        flags = _event_flags(row)
        if not flags["identity_anomaly"]:
            continue
        current = _geometry(row, "sel") if flags["current_complete"] else None
        primary = _geometry(row, "primary") if flags["primary_complete"] else None
        result = {
            "event_id": row.get("event_id", ""),
            "current_complete": int(flags["current_complete"]),
            "primary_complete": int(flags["primary_complete"]),
            "primary_track_id": row.get("primary_track_id", ""),
            "selected_track_ids": ";".join(
                str(_integer(row.get(f"sel_{layer}_track_id")) or "") for layer in LAYERS
            ),
            "selected_parent_ids": ";".join(
                "" if _integer(row.get(f"sel_{layer}_parent_id")) is None
                else str(_integer(row.get(f"sel_{layer}_parent_id")))
                for layer in LAYERS
            ),
            "current_theta_x_mrad": current["theta_x_mrad"] if current else "",
            "current_theta_y_mrad": current["theta_y_mrad"] if current else "",
            "current_theta_mrad": current["theta_mrad"] if current else "",
            "primary_theta_x_mrad": primary["theta_x_mrad"] if primary else "",
            "primary_theta_y_mrad": primary["theta_y_mrad"] if primary else "",
            "primary_theta_mrad": primary["theta_mrad"] if primary else "",
            "delta_theta_mrad": (current["theta_mrad"] - primary["theta_mrad"])
                                if current and primary else "",
            "current_poca_x_mm": current["poca_x_mm"] if current else "",
            "current_poca_y_mm": current["poca_y_mm"] if current else "",
            "current_poca_z_mm": current["poca_z_mm"] if current else "",
            "primary_poca_x_mm": primary["poca_x_mm"] if primary else "",
            "primary_poca_y_mm": primary["poca_y_mm"] if primary else "",
            "primary_poca_z_mm": primary["poca_z_mm"] if primary else "",
            "delta_poca_mm": math.sqrt(sum(
                (current[f"poca_{axis}_mm"] - primary[f"poca_{axis}_mm"]) ** 2
                for axis in ("x", "y", "z")
            )) if current and primary else "",
            "current_theta_above_run_p99_descriptive_reference": int(
                current is not None and p99 is not None and current["theta_mrad"] > p99
            ),
            "run_current_theta_p99_mrad_descriptive_only": p99 if p99 is not None else "",
        }
        rows.append(result)
    return rows, p99


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anomalies-output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    event_rows = _read_csv(args.events)
    result = summarize(event_rows, _read_csv(args.boundaries))
    cases, p99 = anomaly_cases(event_rows)
    result["run_id"] = args.run_id
    result["anomaly_case_count"] = len(cases)
    result["current_complete_theta_p99_mrad_descriptive_only"] = p99
    result["events_csv"] = str(args.events)
    result["boundary_entries_csv"] = str(args.boundaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    anomaly_path = args.anomalies_output
    anomaly_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(cases[0]) if cases else ["event_id", "current_complete", "primary_complete"]
    with anomaly_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)


if __name__ == "__main__":
    main()
