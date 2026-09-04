import importlib.util
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# imports bypassing package init
def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # We don't register these in sys.modules globally under their file name to avoid conflict,
    # or we do but with unique names.
    spec.loader.exec_module(module)
    return module


class TestDelegatedPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_delegated_persistence")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

        # [MOCK] Patch sys.modules to mock heavy dependencies
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "spectramr.infrastructure.services": MagicMock(),
                "spectramr.infrastructure.services.iteration_counter_service": MagicMock(),
                "spectramr.domain.entities.training_session": MagicMock(),
                "torch": MagicMock(),
                "torch.distributions": MagicMock(),
                "torch.distributions.chi2": MagicMock(),
            },
        )
        self.modules_patcher.start()

        # Mock torch package structure deeper if needed (for specific lookups)
        # Note: patch.dict replaces the module entries.
        # If imports happen inside the delegates, they will find these mocks.

        self.checkpoint_service_mock = MagicMock()
        self.checkpoint_service_mock.save_checkpoint.return_value = "mock_path.pth"
        self.checkpoint_service_mock.checkpoint_dir = str(self.test_dir / "checkpoints")

        # Import delegates dynamically AFTER patching
        # We must import them here because they might import the mocked modules at top-level
        # However, the original code used 'import_file' which bypasses normal imports slightly,
        # but the delegates themselves might have standard 'from ... import ...'.

        # We need to ensure the classes are available.
        # Since 'import_file' loads from source, we can call it here.
        # But we need to make sure we don't reload unnecessarily or if we do, it uses the mocked sys.modules.

        self.checkpoint_delegate_cls = import_file(
            "checkpoint_delegate",
            "src/spectramr/infrastructure/services/persistence/checkpoint_delegate.py",
        ).CheckpointDelegate

        # MetricsDelegate deleted with #710: it had no production caller, only
        # this test -- which loaded it BY PATH via `import_file`, so an
        # import-grep could not see the dependency. Worth recording: that is the
        # same dynamic-load pattern that hid seven fake skips in #634.

        self.training_log_delegate_cls = import_file(
            "training_log_delegate",
            "src/spectramr/infrastructure/services/persistence/training_log_delegate.py",
        ).TrainingLogDelegate

        # Instantiate delegates
        self.checkpoint_delegate = self.checkpoint_delegate_cls(
            self.checkpoint_service_mock, output_dir=str(self.test_dir / "checkpoints")
        )
        self.log_delegate = self.training_log_delegate_cls(
            output_dir=str(self.test_dir / "logs")
        )

    def tearDown(self):
        if hasattr(self, "log_delegate"):
            self.log_delegate.shutdown()

        self.modules_patcher.stop()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_checkpoint_delegation(self):
        """Verify checkpoint delegate enforces naming and calls service."""
        session = MagicMock()
        session.config = {"epochs": 100}
        session.model_id = "test_model"
        session.generator = MagicMock()

        self.checkpoint_delegate.save_checkpoint(session, step=1000, epoch=10)

        # Verify call
        self.checkpoint_service_mock.save_checkpoint.assert_called_once()
        call_kwargs = self.checkpoint_service_mock.save_checkpoint.call_args[1]

        # Check file_path instead of name, as delegate constructs it
        self.assertIn("file_path", call_kwargs)
        self.assertTrue(
            call_kwargs["file_path"].endswith("checkpoint_epoch_10_step_1000.pth")
        )
        self.assertEqual(call_kwargs["step"], 1000)


    def test_training_log_delegation(self):
        """Verify training log delegate writes to separate CSV."""
        data = {"loss": 0.5, "lr": 0.001}
        self.log_delegate.log_step(data, step=1, epoch=0)

        # Flush to ensure async write completes
        self.log_delegate.flush()

        log_file = self.test_dir / "logs" / "loss_log.csv"
        self.assertTrue(log_file.exists())

        with open(log_file) as f:
            content = f.read()
            self.assertIn("loss", content)
            self.assertIn("0.5", content)


if __name__ == "__main__":
    unittest.main()
