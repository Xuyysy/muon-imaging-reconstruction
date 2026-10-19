import unittest

from write_manifest import simulation_source_provenance


class ManifestProvenanceTests(unittest.TestCase):
    def test_original_geant4_manifest_remains_authoritative_after_schema_upgrade(self):
        original = {
            "source_commit": "geant4-run-commit",
            "source_file_sha256": {"source.cc": "original-hash"},
        }
        previous = {
            "geant4_source_commit": "wrong-new-analysis-commit",
            "analysis_source_commit": "new-analysis-commit",
            "geant4_source_file_sha256": {"source.cc": "new-analysis-hash"},
        }
        commit, hashes = simulation_source_provenance(previous, original, {"source.cc": "fallback"})
        self.assertEqual(commit, "geant4-run-commit")
        self.assertEqual(hashes, {"source.cc": "original-hash"})


if __name__ == "__main__":
    unittest.main()
