import unittest

from summarize_identity import anomaly_cases, boundary_order_cases, summarize


LAYERS = ("up1", "up2", "down1", "down2")


def event(event_id, selected_ids=(1, 1, 1, 1), primary_id=1,
          selected_parents=(0, 0, 0, 0), current_complete=True,
          primary_complete=True, primary_candidate_count=1,
          coordinate_shift_layer=None):
    row = {
        "event_id": event_id,
        "primary_candidate_count": primary_candidate_count,
        "primary_track_id": primary_id if primary_candidate_count == 1 else "",
        "current_complete": int(current_complete),
        "primary_complete": int(primary_complete),
    }
    for index, layer in enumerate(LAYERS):
        selected_present = current_complete or index < 2
        primary_present = primary_complete or index < 2
        row.update({
            f"sel_{layer}_track_id": selected_ids[index] if selected_present else "",
            f"sel_{layer}_pdg_id": 13 if selected_present else "",
            f"sel_{layer}_parent_id": selected_parents[index] if selected_present else "",
            f"sel_{layer}_global_time_ns": float(index) if selected_present else "",
            f"sel_{layer}_x_mm": float(index) if selected_present else "",
            f"sel_{layer}_y_mm": 0.0 if selected_present else "",
            f"sel_{layer}_z_mm": float(index * 10) if selected_present else "",
            f"primary_{layer}_track_id": primary_id if primary_present else "",
            f"primary_{layer}_pdg_id": 13 if primary_present else "",
            f"primary_{layer}_parent_id": 0 if primary_present else "",
            f"primary_{layer}_global_time_ns": float(index) if primary_present else "",
            f"primary_{layer}_x_mm": (
                float(index) + (1.0 if coordinate_shift_layer == layer else 0.0)
                if primary_present else ""
            ),
            f"primary_{layer}_y_mm": 0.0 if primary_present else "",
            f"primary_{layer}_z_mm": float(index * 10) if primary_present else "",
        })
    return row


class SummaryTests(unittest.TestCase):
    def test_counts_generated_and_complete_events_without_losing_incomplete_events(self):
        result = summarize([
            event(0),
            event(1, current_complete=False, primary_complete=False),
        ], [])
        self.assertEqual(result["generated_events"], 2)
        self.assertEqual(result["current_complete_events"], 1)
        self.assertEqual(result["primary_complete_events"], 1)

    def test_reports_overlapping_mixed_track_and_secondary_categories_separately(self):
        result = summarize([
            event(0, selected_ids=(1, 2, 1, 1), selected_parents=(0, 1, 0, 0)),
        ], [])
        self.assertEqual(result["track_id_inconsistent_events"], 1)
        self.assertEqual(result["selected_secondary_events"], 1)
        self.assertEqual(result["all_four_current_from_primary_muon_events"], 0)
        self.assertEqual(result["identity_anomaly"]["count"], 1)

    def test_single_secondary_track_is_an_anomaly_even_when_track_ids_match_each_other(self):
        result = summarize([
            event(0, selected_ids=(2, 2, 2, 2), selected_parents=(1, 1, 1, 1)),
        ], [])
        self.assertEqual(result["track_id_inconsistent_events"], 0)
        self.assertEqual(result["selected_secondary_events"], 1)
        self.assertEqual(result["identity_anomaly"]["count"], 1)

    def test_unidentifiable_primary_is_reported_separately_from_known_anomaly(self):
        result = summarize([
            event(0, primary_candidate_count=0, primary_id=""),
        ], [])
        self.assertEqual(result["primary_identity_unknown_events"], 1)
        self.assertEqual(result["identity_anomaly"]["denominator"], 0)

    def test_primary_identity_counts_are_mutually_exclusive_missing_unique_ambiguous(self):
        result = summarize([
            event(0, primary_candidate_count=0, primary_id=""),
            event(1),
            event(2, primary_candidate_count=2, primary_id=""),
        ], [])
        self.assertEqual(result["primary_identity_missing_events"], 1)
        self.assertEqual(result["primary_identity_unique_events"], 1)
        self.assertEqual(result["primary_identity_ambiguous_events"], 1)
        self.assertEqual(result["primary_identity_unknown_events"], 2)

    def test_coordinate_difference_only_uses_events_complete_under_both_rules(self):
        result = summarize([
            event(0, coordinate_shift_layer="down1"),
            event(1, primary_complete=False),
        ], [])
        self.assertEqual(result["coordinate_comparable_events"], 1)
        self.assertEqual(result["current_vs_primary_coordinates_differ_events"], 1)

    def test_acceptance_set_difference_compares_both_rules_over_all_events(self):
        result = summarize([
            event(0),
            event(1, current_complete=True, primary_complete=False),
            event(2, current_complete=False, primary_complete=True),
        ], [])
        self.assertEqual(result["acceptance_set_difference_events"], 2)

    def test_zero_anomalies_has_nonzero_wilson_upper_bound(self):
        result = summarize([event(i) for i in range(10)], [])
        interval = result["identity_anomaly"]["wilson_95_interval"]
        self.assertEqual(result["identity_anomaly"]["count"], 0)
        self.assertGreater(interval["upper"], 0.0)
        self.assertLess(interval["upper"], 1.0)

    def test_processing_order_is_compared_with_minimum_global_time(self):
        selected = event(0)
        selected["sel_up1_global_time_ns"] = 8.0
        entries = [
            {"event_id": 0, "detector": "UpstreamDetector1", "processing_order": 1, "global_time_ns": 8.0},
            {"event_id": 0, "detector": "UpstreamDetector1", "processing_order": 2, "global_time_ns": 5.0},
        ]
        result = summarize([selected], entries)
        self.assertEqual(result["events_with_processing_vs_earliest_time_difference"], 1)
        self.assertEqual(result["layers_with_processing_vs_earliest_time_difference"], 1)

    def test_anomaly_geometry_reports_angle_and_poca_differences(self):
        rows, p99 = anomaly_cases([
            event(0, selected_ids=(1, 2, 1, 1), selected_parents=(0, 1, 0, 0),
                  coordinate_shift_layer="down1"),
        ])
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["delta_theta_mrad"], 0.0)
        self.assertGreater(rows[0]["delta_poca_mm"], 0.0)
        self.assertIsNotNone(p99)

    def test_anomaly_uses_unmodified_cpp_track_output_for_current_fit(self):
        event_row = event(0, selected_ids=(1, 2, 1, 1), selected_parents=(0, 1, 0, 0),
                          coordinate_shift_layer="down1")
        tracks = [{
            "event": "0", "theta_x_mrad": "12.345", "theta_y_mrad": "-6.789",
            "theta_mrad": "14.087", "poca_x_mm": "2.5", "poca_y_mm": "-3.0",
            "poca_z_mm": "9.25",
        }]
        rows, _ = anomaly_cases([event_row], tracks)
        self.assertEqual(rows[0]["current_theta_mrad"], 14.087)
        self.assertEqual(rows[0]["current_poca_z_mm"], 9.25)

    def test_boundary_order_case_identifies_processing_and_earliest_tracks(self):
        selected = event(0)
        selected["sel_up1_track_id"] = "1"
        selected["sel_up1_global_time_ns"] = "8.0"
        entries = [
            {"event_id": 0, "detector": "UpstreamDetector1", "processing_order": 1,
             "global_time_ns": 8.0, "track_id": 1, "pdg_id": 13, "parent_id": 0,
             "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0},
            {"event_id": 0, "detector": "UpstreamDetector1", "processing_order": 2,
             "global_time_ns": 5.0, "track_id": 2, "pdg_id": 11, "parent_id": 1,
             "x_mm": 1.0, "y_mm": 0.0, "z_mm": 0.0},
        ]
        cases = boundary_order_cases([selected], entries)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["first_processed_track_id"], 1)
        self.assertEqual(cases[0]["earliest_time_track_id"], 2)
        self.assertEqual(cases[0]["earliest_time_parent_id"], 1)
        self.assertAlmostEqual(cases[0]["selected_vs_earliest_time_delta_ns"], 3.0)


if __name__ == "__main__":
    unittest.main()
