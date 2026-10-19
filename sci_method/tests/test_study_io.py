import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.study_io import create_output_directory  # noqa: E402


class StudyOutputSafetyTests(unittest.TestCase):
    def test_existing_result_path_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run-001"
            path.mkdir()
            marker = path / "evidence.json"
            marker.write_text("preserve-me")
            with self.assertRaises(FileExistsError):
                create_output_directory(path)
            self.assertEqual(marker.read_text(), "preserve-me")

    def test_creates_new_study_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = create_output_directory(Path(temp_dir) / "new" / "run-001")
            self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
