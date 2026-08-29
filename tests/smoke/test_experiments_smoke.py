import glob
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

# Imports moved to test scope


# Timeout handler
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# Discover config files
CONFIG_DIR = Path("experiments/validated")
CONFIG_FILES = glob.glob(str(CONFIG_DIR / "dummy_*.yaml"))


def verify_specific_fixes(settings):
    """Integrate logic from verify_fixes.py"""
    # K-Space Fix Verification
    t_config = getattr(settings, "training", None)
    if t_config:
        # If it's the K-Space experiment, it MUST be 'kspace'
        task = getattr(t_config, "task", "")
        if "kspace" in str(task) or getattr(settings.model, "target_domain", "") == "kspace":
            # Assert the NESTED domain (the field strategies actually read), not
            # the optional top-level convenience knob. Top-level model_domain
            # defaults to None ("defer to nested") so K-space configs that set
            # only model.model_domain leave it unset — see settings.py and the
            # apply_overrides round-trip fix (2026-05-29).
            nested_domain = getattr(settings.model, "model_domain", None) or getattr(
                settings.model, "target_domain", None
            )
            assert nested_domain == "kspace", (
                "model.model_domain/target_domain must be 'kspace' for K-Space experiments"
            )

    # VAE Fix Verification
    m_config = getattr(settings, "model", None)
    if (
        m_config
        and m_config.model_type == "vae_3d"
        and "multicontrast" in getattr(settings.metadata, "name", "")
    ):
        assert m_config.in_channels == 1, "VAE 32a must have in_channels=1"


@pytest.mark.parametrize("config_path", CONFIG_FILES)
def test_experiment_smoke(config_path, tmp_path):
    """
    Smoke test for experiment configurations.
    1. Loads config
    2. Verifies specific regression fixes
    3. runs training loop for 2 iterations (dry run)
    """
    print(f"\nTesting config: {config_path}")

    # These imports were wrapped in ``patch.dict(sys.modules, {"torchkbnuf":
    # MagicMock()})`` as an "[ENV FIX] Mock dependencies that might be missing".
    # ``torchkbnuf`` is not a real module name (the package is ``torchkbnufft``,
    # a hard dependency that is installed), so nothing was ever mocked -- while
    # the ``patch.dict`` teardown evicted the two modules imported inside it,
    # forcing a re-import that re-ran every ``@register_model`` decorator. See
    # ``tests/smoke/test_active_experiments.py``.
    from mriforge.config.settings import TrainingSettings
    from mriforge.pipelines import run_training_pipeline

    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # 1. Load Settings
    settings = TrainingSettings.from_yaml(config_path)

    # 2. Verify Fixes
    verify_specific_fixes(settings)

    # Helpers to safe access
    def get_attr(obj, attr, default=None):
        return getattr(obj, attr, default)

    # 3. Configure for Smoke Test (Fast Run) using model_copy for frozen models

    # Helper to safe update
    def safe_update(obj, overrides):
        if obj is None:
            return None
        if hasattr(obj, "model_copy"):
            return obj.model_copy(update=overrides)
        if isinstance(obj, dict):
            new_obj = obj.copy()
            new_obj.update(overrides)
            return new_obj
        return obj

    # Update Logging
    new_logging = safe_update(
        get_attr(settings, "logging"),
        {
            "silent": True,
            "log_to_file": False,
            "log_to_console": False,
            "enable_experiment_tracking": False,
            "tracking_service": "none",
            "log_dir": str(tmp_path / "logs"),
        },
    )

    # Update Checkpoint
    new_checkpoint = safe_update(get_attr(settings, "checkpoint"), {"enabled": False})

    # Update Metrics (if exists)
    new_metrics = safe_update(get_attr(settings, "metrics"), {"enable_tracking": False})

    # Update Validation
    new_validation = safe_update(get_attr(settings, "validation"), {"enable_visualization": True})

    # Construction update dict for top-level settings (Handling new schema)
    # We must check if 'training' exists and update it, otherwise safe loop won't terminate

    # 1. Update Training Config (max_iterations, etc for the new schema)
    new_training = None
    t_config = get_attr(settings, "training")
    if t_config:
        new_training = t_config.model_copy(
            update={"max_iterations": 2, "epochs": 1, "max_steps": 2}
        )
    else:
        # Fallback for old configs if any (though migration should have happened)
        pass

    updates = {"logging": new_logging, "loss_logging": {"enabled": False}}

    if new_training:
        updates["training"] = new_training

    # Legacy fallback (if top level properties exist)
    # updates["max_iterations"] = 2  # Disabling legacy to avoid potential issues

    if new_checkpoint:
        updates["checkpoint"] = new_checkpoint
    if new_metrics:
        updates["metrics"] = new_metrics
    if new_validation:
        updates["validation"] = new_validation

    # [GPU FIX] Enable GPU for Smoke Test if available
    if torch.cuda.is_available():
        # Handle acceleration specifically - it might be None
        acc_dict = {"device": "cuda", "use_amp": True, "multi_gpu": False}
        acc_config = get_attr(settings, "acceleration")
        if acc_config:
            updates["acceleration"] = acc_config.model_copy(update=acc_dict)
        else:
            updates["acceleration"] = acc_dict

        # Also ensure training uses the correct device
        if new_training:
            updates["training"] = new_training.model_copy(update={"device": "cuda"})
    else:
        # Explicitly set to cpu if no cuda to avoid ambiguity
        acc_dict = {"device": "cpu", "use_amp": False, "multi_gpu": False}
        acc_config = get_attr(settings, "acceleration")
        if acc_config:
            updates["acceleration"] = acc_config.model_copy(update=acc_dict)
        else:
            updates["acceleration"] = acc_dict

    # Update Data (Patch Size for Dummy Data Compatibility)
    d_config = get_attr(settings, "data")
    if d_config:
        # Determine Patch Size
        new_patch = [32, 32, 1]

        # Check model type for VAE/3D patch adjustment
        # We need to check model type BEFORE settings recreation to set patch size correctly
        m_type_check = "unknown"
        m_config = get_attr(settings, "model")
        if m_config:
            m_type_check = get_attr(m_config, "model_type", get_attr(m_config, "type", "unknown"))

        if "vae" in str(m_type_check) or "3d" in str(m_type_check):
            # [FIX] Data is 2D (D=1), so patch must satisfy D<=1.
            # The VAE/LDM has internal logic to repeat D to 16 if input is small.
            new_patch = [32, 32, 1]

        data_overrides = {"patch_size": new_patch, "batch_size": 1, "pin_memory": False}

        # [FIX] Overrides for Exp 32a, 32b and graph diffusion to ensure dummy data is found
        if (
            "experiment_32b_ldm" in config_path
            or "experiment_32a_vae" in config_path
            or "graph_cold_diffusion" in config_path
        ):
            data_overrides["dataset_type"] = "fastmri_knee"
            data_overrides["index_path"] = "data/manifests/fastmri_knee_singlecoil.pkl"
            data_overrides["data_root"] = "databases/fastmri/datasets/knee_singlecoil_train"

        new_data = safe_update(d_config, data_overrides)

        if isinstance(new_data, dict) and "modalities" in new_data:
            del new_data["modalities"]

        updates["data"] = new_data

    # Apply updates to Create New Settings Object
    settings = settings.model_copy(update=updates)

    # [FIX] Exp 32b LDM expects 1 channel, but FastMRI Dummy is 2 (Complex).
    # Since we use FastMRI for smoke test, we see 2 channels input.
    # We must update model config to accept 2 channels for the smoke test valid run.
    if "experiment_32b_ldm" in config_path:
        m_config = getattr(settings, "model", None)
        if m_config:
            # Update top-level in_channels if present
            if hasattr(m_config, "in_channels"):
                # We need to re-create model config
                new_model = m_config.model_copy(update={"in_channels": 2})
                settings = settings.model_copy(update={"model": new_model})

    # Update Model Image Size if needed (after copy, or during if logic requires)
    # We need to dig into model_kwargs
    m_config = getattr(settings, "model", None)
    if m_config and hasattr(m_config, "model_kwargs"):
        kwargs = m_config.model_kwargs.copy()
        modified = False

        # Also check model_kwargs for in_channels usage (some models might duplicate it)
        if "experiment_32b_ldm" in config_path:
            if "in_channels" in kwargs:
                kwargs["in_channels"] = 2
                modified = True
            # [FIX] Also override out_channels to match in_channels (2) for prediction consistency in smoke test
            if "out_channels" in kwargs:
                kwargs["out_channels"] = 2
                modified = True
            # If out_channels not in kwargs, Generator usually defaults to in_channels, which we set to 2.
            # But explicit is better if key exists.

        if "image_size" in kwargs:
            kwargs["image_size"] = [32, 32]
            modified = True

        if "resolution" in kwargs:
            kwargs["resolution"] = [32, 32]
            modified = True

        # Handle potential typo: use 'model_type' if 'type' is missing
        m_type = getattr(m_config, "model_type", getattr(m_config, "type", "unknown"))

        # FNO Specific Reductions (Check model type or keys)
        if str(m_type) == "fno" or "modes" in kwargs or "modes1" in kwargs:
            # Reduce modes
            if "modes" in kwargs:
                modes = kwargs["modes"]
                if isinstance(modes, (list, tuple)):
                    kwargs["modes"] = [min(m, 8) for m in modes]
                    modified = True

            if "modes1" in kwargs:
                kwargs["modes1"] = min(kwargs["modes1"], 8)
                modified = True

            if "modes2" in kwargs:
                kwargs["modes2"] = min(kwargs["modes2"], 8)
                modified = True

            # Cap width/hidden_channels
            if "width" in kwargs:
                kwargs["width"] = 8
                modified = True
            if "hidden_channels" in kwargs:
                kwargs["hidden_channels"] = 8
                modified = True

        if modified:
            # Need to recreate model settings since it might be frozen
            new_model = m_config.model_copy(update={"model_kwargs": kwargs})
            settings = settings.model_copy(update={"model": new_model})
            m_config = new_model  # Update m_config reference to the new model

        if "experiment_32a" in config_path:
            # Force Data to be 1 channel (Magnitude) to match VAE config
            d_config = getattr(settings, "data", None)
            if d_config:
                new_data = d_config.model_copy(
                    update={"return_image_domain": True}
                )  # Returns Magnitude (1ch)
                settings = settings.model_copy(update={"data": new_data})

            # Still valid to enforce model channels if needed, but data match is cleaner
            # Revert forcing model to 2 channels, assume 1 is correct if data provides 1.
            pass

        # [FIX] VAE OOM Fix for Smoke Test
        m_type_check = getattr(m_config, "model_type", getattr(m_config, "type", "unknown"))
        is_vae = "vae" in str(m_type_check) or "vae" in config_path
        if is_vae and m_config:
            # Reduce base_channels to 16
            if hasattr(m_config, "base_channels"):
                new_model = m_config.model_copy(update={"base_channels": 16})
                settings = settings.model_copy(update={"model": new_model})
                m_config = new_model

            if hasattr(m_config, "model_kwargs") and "base_channels" in m_config.model_kwargs:
                kwargs = m_config.model_kwargs.copy()
                kwargs["base_channels"] = 16
                # [FIX] Override input_shape to match patch size to avoid OOM during feature size calculation
                # The VAE runs a dummy forward pass with input_shape at init
                if "input_shape" in kwargs:
                    kwargs["input_shape"] = [32, 32, 16]

                # [FIX] FastMRI is 2 channels (Complex), VAE expects 1 by default. Override.
                kwargs["in_channels"] = 2
                kwargs["out_channels"] = 2

                # Also update the top-level model.in_channels and out_channels field
                new_model = m_config.model_copy(
                    update={"model_kwargs": kwargs, "in_channels": 2, "out_channels": 2}
                )
                settings = settings.model_copy(update={"model": new_model})
                m_config = new_model  # Update m_config reference to the new model

    # Enable dry execution if possible, but run_training_pipeline is what we strictly test
    # We rely on max_iterations=2 to exit early.

    # 4. Execute with Timeout
    try:
        with time_limit(600):  # [FIX] Increase to 600s for GPU initialization
            # DEBUG PRINT
            m_config = getattr(settings, "model", None)
            if m_config and hasattr(m_config, "model_kwargs"):
                print(f"\n[DEBUG-SMOKE] Settings model_kwargs: {m_config.model_kwargs}")

            # We need to handle potential sys.exit() if the pipeline fails hard
            # But normally run_training_pipeline raises exceptions
            result = run_training_pipeline(settings)
            assert result is not None
            if isinstance(result, dict) and "error" in result:
                err = str(result["error"])
                # Skip — not fail — when the failure is purely "data file
                # missing on this machine". These dummy YAMLs reference
                # M4Raw paths that only exist on the cluster; the test is
                # otherwise a useful gating signal so we keep it in CI but
                # don't penalise local runs that lack the dataset.
                missing_data_markers = (
                    "No such file or no access",
                    "Unable to synchronously open file",
                    "errno = 2",
                    "[Errno 2]",
                    "Index file not found",
                    "manifest_path",
                )
                if any(m in err for m in missing_data_markers):
                    pytest.skip(f"Required dataset not present locally: {err.splitlines()[0]}")
                pytest.fail(f"Pipeline failed with error: {err}")

    except TimeoutError:
        pytest.fail(f"Experiment {config_path} timed out after 600s")
    except Exception as e:
        # We want to report the error, typically these are real failures we want to catch
        import traceback

        error_trace = traceback.format_exc()
        pytest.fail(f"Experiment {config_path} failed: {e}\nTraceback:\n{error_trace}")


if __name__ == "__main__":
    # Allow running directly for debug
    pytest.main([__file__])
