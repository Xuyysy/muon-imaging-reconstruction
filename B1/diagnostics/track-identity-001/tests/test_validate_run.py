import unittest

from validate_run import validate_run


class ValidateRunTests(unittest.TestCase):
    def test_accepts_consistent_run_rows_and_geant4_count(self):
        rows = [{
            "event_id": "0", "primary_candidate_count": "1", "primary_track_id": "7",
            "primary_pdg_id": "13", "primary_parent_id": "0", "primary_global_time_ns": "0",
            "current_complete": "0",
        }]
        summary = {"generated_events": 1, "current_complete_events": 0}
        self.assertEqual(validate_run(
            rows, summary,
            "[Muon source] IDEAL mode: fixed 4 GeV, +z direction\nComplete tracks: 0",
            1,
        ), [])

    def test_rejects_missing_primary_candidate_and_count_disagreement(self):
        rows = [{"event_id": "0", "primary_candidate_count": "0", "current_complete": "0"}]
        summary = {"generated_events": 1, "current_complete_events": 0}
        errors = validate_run(
            rows, summary,
            "[Muon source] IDEAL mode: fixed 4 GeV, +z direction\nComplete tracks: 1",
            1,
        )
        self.assertTrue(any("primary mu- candidates" in error for error in errors))
        self.assertTrue(any("RunAction" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
