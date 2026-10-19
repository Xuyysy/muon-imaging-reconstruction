import sys
import inspect
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.data_contract import (  # noqa: E402
    HIT_COLUMNS,
    ObservedEvent,
    observed_event_from_mapping,
    observed_events_from_rows,
)
from sci_method.forward_model import gaussian_nll_and_gradient  # noqa: E402


def make_row():
    row = {"event": "7"}
    row.update({column: str(index + 0.25) for index, column in enumerate(HIT_COLUMNS)})
    return row


class ObservedDataContractTests(unittest.TestCase):
    def test_keeps_four_hits_in_declared_plane_order_and_ignores_derived_fields(self):
        row = make_row()
        row.update({"poca_x_mm": "999", "theta_mrad": "999", "truth": "target"})

        event = observed_event_from_mapping(row, run_id="geant4-run-A")

        self.assertEqual(event, ObservedEvent("geant4-run-A", 7, ((0.25, 1.25, 2.25),
                                                (3.25, 4.25, 5.25),
                                                (6.25, 7.25, 8.25),
                                                (9.25, 10.25, 11.25))))
        self.assertFalse(hasattr(event, "poca_x_mm"))
        self.assertFalse(hasattr(event, "truth"))
        self.assertEqual(event.key, ("geant4-run-A", 7))

    def test_does_not_mutate_input_row(self):
        row = make_row()
        original = row.copy()

        observed_event_from_mapping(row, run_id="geant4-run-A")

        self.assertEqual(row, original)

    def test_reports_missing_observed_fields(self):
        row = make_row()
        del row[HIT_COLUMNS[-1]]

        with self.assertRaisesRegex(ValueError, "missing observed input columns"):
            observed_event_from_mapping(row, run_id="geant4-run-A")

    def test_rejects_non_finite_hit_coordinates(self):
        row = make_row()
        row[HIT_COLUMNS[0]] = "nan"

        with self.assertRaisesRegex(ValueError, "must be finite"):
            observed_event_from_mapping(row, run_id="geant4-run-A")

    def test_batch_conversion_requires_run_id_and_creates_composite_keys(self):
        rows = [make_row(), make_row()]
        rows[1]["event"] = "8"

        events = observed_events_from_rows(rows, run_id="geant4-run-A")

        self.assertEqual(tuple(event.event_id for event in events), (7, 8))
        self.assertEqual(tuple(event.key for event in events), (("geant4-run-A", 7), ("geant4-run-A", 8)))
        self.assertIsInstance(events, tuple)
        self.assertTrue(all(isinstance(event, ObservedEvent) for event in events))

    def test_rejects_duplicate_composite_event_keys_in_one_batch(self):
        with self.assertRaisesRegex(ValueError, "duplicate event key"):
            observed_events_from_rows([make_row(), make_row()], run_id="geant4-run-A")

    def test_same_event_number_in_different_runs_has_distinct_identity(self):
        row = make_row()
        first = observed_event_from_mapping(row, run_id="geant4-run-A")
        second = observed_event_from_mapping(row, run_id="geant4-run-B")
        self.assertEqual(first.event_id, second.event_id)
        self.assertNotEqual(first.key, second.key)

    def test_rejects_fractional_and_boolean_event_ids_instead_of_truncating(self):
        for invalid in ("7.5", 7.5, True):
            row = make_row()
            row["event"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "integer"):
                observed_event_from_mapping(row, run_id="geant4-run-A")

    def test_requires_nonempty_run_id(self):
        with self.assertRaisesRegex(ValueError, "run_id"):
            observed_event_from_mapping(make_row(), run_id=" ")

    def test_inference_input_contract_excludes_truth_and_oracle_fields(self):
        self.assertEqual(set(ObservedEvent.__dataclass_fields__), {"run_id", "event_id", "hits_mm"})
        self.assertEqual(
            set(inspect.signature(gaussian_nll_and_gradient).parameters),
            {"residuals", "detector_covariances", "kernels", "density_mm_inv"},
        )


if __name__ == "__main__":
    unittest.main()
