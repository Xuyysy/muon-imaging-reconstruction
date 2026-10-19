import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.data_contract import (  # noqa: E402
    HIT_COLUMNS,
    ObservedEvent,
    observed_event_from_mapping,
    observed_events_from_rows,
)


def make_row():
    row = {"event": "7"}
    row.update({column: str(index + 0.25) for index, column in enumerate(HIT_COLUMNS)})
    return row


class ObservedDataContractTests(unittest.TestCase):
    def test_keeps_four_hits_in_declared_plane_order_and_ignores_derived_fields(self):
        row = make_row()
        row.update({"poca_x_mm": "999", "theta_mrad": "999", "truth": "target"})

        event = observed_event_from_mapping(row)

        self.assertEqual(event, ObservedEvent(7, ((0.25, 1.25, 2.25),
                                                (3.25, 4.25, 5.25),
                                                (6.25, 7.25, 8.25),
                                                (9.25, 10.25, 11.25))))
        self.assertFalse(hasattr(event, "poca_x_mm"))
        self.assertFalse(hasattr(event, "truth"))

    def test_does_not_mutate_input_row(self):
        row = make_row()
        original = row.copy()

        observed_event_from_mapping(row)

        self.assertEqual(row, original)

    def test_reports_missing_observed_fields(self):
        row = make_row()
        del row[HIT_COLUMNS[-1]]

        with self.assertRaisesRegex(ValueError, "missing observed input columns"):
            observed_event_from_mapping(row)

    def test_rejects_non_finite_hit_coordinates(self):
        row = make_row()
        row[HIT_COLUMNS[0]] = "nan"

        with self.assertRaisesRegex(ValueError, "must be finite"):
            observed_event_from_mapping(row)

    def test_batch_conversion_is_immutable_and_allows_reused_event_ids(self):
        rows = [make_row(), make_row()]

        events = observed_events_from_rows(rows)

        self.assertEqual(tuple(event.event_id for event in events), (7, 7))
        self.assertIsInstance(events, tuple)
        self.assertTrue(all(isinstance(event, ObservedEvent) for event in events))


if __name__ == "__main__":
    unittest.main()
