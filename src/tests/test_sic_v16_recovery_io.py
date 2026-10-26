"""Standard-library-only recovery checks, with no scientific/model imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load("v16_recovery_io", ROOT / "src/eegproc/deep_learning/joint_architectures/SICModelv16/recovery.py")
storage = load("v16_storage_io", ROOT / "experiments/sic_v16/storage.py")


class RecoveryIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-artifacts").mkdir(exist_ok=True)

    def test_progress_ignores_partial_files_and_preserves_them(self):
        with tempfile.TemporaryDirectory(prefix="sicv16-recovery-", dir=ROOT / ".test-artifacts") as folder:
            assert Path(folder).resolve().is_relative_to((ROOT / ".test-artifacts").resolve())
            root = Path(folder)
            subject = root / "subject_00"
            subject.mkdir()
            saved = subject / "epoch_001.keras"
            with zipfile.ZipFile(saved, "w") as archive:
                archive.writestr("config.json", "{}")
                archive.writestr("model.weights.h5", "synthetic fixture")
            record = {"target_subject": 0, "epoch": 1,
                      "checkpoint": "subject_00/epoch_001.keras"}
            (subject / "epoch_001.json").write_text(json.dumps(record))
            partial = subject / "epoch_002.tmp.keras"
            partial.write_text("interrupted save")
            progress = recovery.checkpoint_progress(root, 0, 50, verify_latest=True)
            self.assertEqual(progress["completed_epochs"], 1)
            self.assertFalse(progress["complete"])
            self.assertTrue(partial.exists())
            saved.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                recovery.checkpoint_progress(root, 0, 50)

    def test_migration_checks_bytes_before_removing_source(self):
        with tempfile.TemporaryDirectory(prefix="sicv16-recovery-", dir=ROOT / ".test-artifacts") as folder:
            assert Path(folder).resolve().is_relative_to((ROOT / ".test-artifacts").resolve())
            source = Path(folder) / "home_runs"
            source.mkdir()
            (source / "checkpoint.keras").write_bytes(b"preserve all checkpoint bytes")
            destination = Path(folder) / "work_runs"
            # Windows may not permit symlink creation. Exercise the verified
            # copy/removal stage independently of OS-specific link privileges.
            with patch.object(Path, "symlink_to"):
                storage.migrate_store(source, destination)
            self.assertEqual((destination / "checkpoint.keras").read_bytes(), b"preserve all checkpoint bytes")
            self.assertFalse(source.exists())

    def test_migration_conflict_keeps_original(self):
        with tempfile.TemporaryDirectory(prefix="sicv16-recovery-", dir=ROOT / ".test-artifacts") as folder:
            assert Path(folder).resolve().is_relative_to((ROOT / ".test-artifacts").resolve())
            source, destination = Path(folder)/"home_runs", Path(folder)/"work_runs"
            source.mkdir()
            destination.mkdir()
            (source/"epoch.keras").write_bytes(b"original")
            (destination/"epoch.keras").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                storage.migrate_store(source, destination)
            self.assertEqual((source/"epoch.keras").read_bytes(), b"original")
            self.assertEqual((destination/"epoch.keras").read_bytes(), b"different")


if __name__ == "__main__":
    unittest.main(verbosity=2)
