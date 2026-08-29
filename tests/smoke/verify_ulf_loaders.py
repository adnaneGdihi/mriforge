import sys
import tracemalloc

tracemalloc.start()
from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.builders.context import BuilderContext
from mriforge.infrastructure.builders.directors.data_pipeline_director import DataPipelineDirector

ulf_targets = [
    "experiments/inprogress/hilbert_mamba/exp_hm_01_ct_mamba.yaml",
    "experiments/inprogress/hilbert_mamba/exp_hm_08_sdtw_mamba.yaml",
    "experiments/inprogress/10_paradigms/exp_08_structure_tensor_transformer.yaml",
    "experiments/inprogress/diffusion/experiment_32b_ldm_ulf_to_hf.yaml",
    "experiments/inprogress/experiment_53_guided_ulf_to_hf_sr.yaml",
    "experiments/inprogress/vae_latent/experiment_32a_vae_ulf_to_hf_paired_CORRECTED.yaml",
]

all_passed = True
for fpath in ulf_targets:
    print(f"\n--- Testing {fpath} ---")
    try:
        config = TrainingSettings.from_yaml(fpath)
        train_loader, val_loader = DataPipelineDirector(BuilderContext(config=config)).build_dataloaders()
        print(f" [OK] DataLoaders built. Train pairs: {len(train_loader.dataset)} | Val pairs: {len(val_loader.dataset)}")

    except Exception as e:
        print(f" [FAIL] {type(e).__name__}: {str(e)}")
        all_passed = False

if all_passed:
    sys.exit(0)
else:
    sys.exit(1)
