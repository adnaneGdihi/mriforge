.. _tutorial_06_hyperparameter_optimization:

================================================
Tutorial 06: Hyperparameter Optimization (HPO)
================================================

**Difficulty:** Advanced | **Time:** ~3 hours | **GPU:** 8GB+ VRAM

This tutorial covers the HPO pipeline built on **Optuna**, accessed
through the framework's ``HPOUseCase``. You will run a Bayesian
optimization search over learning rate, model architecture, and loss
weights — and automatically select the best configuration.

.. contents:: Table of Contents
   :local:
   :depth: 2


Prerequisites
=============

- Completed :doc:`tutorial_01_basic_reconstruction`
- Familiarity with experiment YAML configs (:doc:`../config_schema_reference`)
- ``optuna`` installed: ``pip install optuna optuna-dashboard``


HPO Architecture
================

.. mermaid::

   flowchart TD
       CLI["src/main.py hpo"] --> UC["HPOUseCase"]
       UC --> HC["HPOCoordinator"]
       HC --> OS["Optuna Study"]
       OS --> T1["Trial 1\n(sampled config)"]
       OS --> T2["Trial 2\n(sampled config)"]
       OS --> TN["Trial N\n..."]
       T1 --> TS["Training Loop\n(truncated)"]
       TS --> OBJ["Objective Value\n(val_psnr @ 5k steps)"]
       OBJ --> OS
       OS --> BEST["Best Trial\n→ Full Training"]

       style UC fill:#4a90d9,color:white
       style OS fill:#7bc47f,color:white
       style BEST fill:#e67e22,color:white


Step 1 — Define the HPO Search Space
=======================================

Create ``experiments/hpo/hpo_reconstruction.yaml``:

.. code-block:: yaml

   experiment_name: hpo_reconstruction_tutorial
   config_version: "6.0"
   device: cuda
   seed: 42

   # HPO-specific section
   hpo:
     n_trials: 30               # Number of Optuna trials
     timeout_hours: 4           # Max wall time
     sampler: tpe               # tpe | random | cmaes | grid
     pruner: median             # median | hyperband | none
     direction: maximize        # maximize val_psnr
     objective_metric: val_psnr
     objective_steps: 5000      # Evaluate at this step (not full training)
     storage: sqlite:///hpo_results.db   # Persistent storage
     study_name: tutorial_06_hpo

     # Define search space
     search_space:
       optimization.learning_rate:
         type: float
         low: 1.0e-5
         high: 1.0e-3
         log: true            # Log-uniform sampling

       optimization.optimizer_type:
         type: categorical
         choices: [adam, adamw]

       optimization.weight_decay:
         type: float
         low: 1.0e-6
         high: 1.0e-2
         log: true

       model.model_type:
         type: categorical
         choices: [standard_unet, enhanced_unet, swin_unet]

       model.model_kwargs.base_channels:
         type: int
         low: 32
         high: 128
         step: 32             # 32, 64, 96, 128

       # Loss weights
       losses_image_l1_weight:          # Maps to losses.image_losses[l1].weight
         type: float
         low: 1.0
         high: 20.0

       losses_image_ssim_weight:
         type: float
         low: 0.0
         high: 5.0

   # Base config (overridden by search space per trial)
   data:
     data_root: databases/fastmri/datasets/knee_singlecoil_train/
     dataset_type: fastmri_knee
     batch_size: 8
     num_workers: 4
     in_channels: 1
     out_channels: 1
     coil_processing_mode: rss

   training:
     training_mode: reconstruction
     max_iterations: 50000        # Full training (used for best trial only)

   losses:
     output_domain: image
     image_losses:
       - name: l1
         weight: 10.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true

   optimization:
     learning_rate: 1e-4
     optimizer_type: adamw
     weight_decay: 1e-4
     lr_scheduler: cosine
     warmup_iterations: 500
     use_amp: true
     gradient_clip_val: 1.0

   checkpoint:
     checkpoint_dir: checkpoints/hpo_trials
     save_interval: 999999     # Don't save during trials (space)
     format: safetensors

   validation:
     enabled: true
     eval_interval: 1000

   physics:
     data_consistency:
       enabled: true
       method: hard


Step 2 — Run the HPO Search
==============================

.. code-block:: bash

   # Launch HPO (runs n_trials × objective_steps iterations)
   python src/main.py hpo \
       --config experiments/hpo/hpo_reconstruction.yaml

   # Parallel workers on same machine (4 GPUs)
   for GPU in 0 1 2 3; do
       CUDA_VISIBLE_DEVICES=$GPU python src/main.py hpo \
           --config experiments/hpo/hpo_reconstruction.yaml \
           --worker-id $GPU &
   done
   wait

   # Monitor live with Optuna Dashboard
   optuna-dashboard sqlite:///hpo_results.db

The dashboard shows trial history, parameter importances, and
Pareto fronts at ``http://localhost:8080``.


Step 3 — Inspect Results Programmatically
==========================================

.. code-block:: python

   import optuna

   study = optuna.load_study(
       study_name="tutorial_06_hpo",
       storage="sqlite:///hpo_results.db",
   )

   # Best trial
   best = study.best_trial
   print(f"Best val_psnr: {best.value:.2f} dB")
   print(f"Best params:   {best.params}")

   # Parameter importance (requires 20+ trials)
   importances = optuna.importance.get_param_importances(study)
   for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
       print(f"  {param:<45} {importance:.3f}")

Expected output:

.. code-block:: text

   Best val_psnr: 36.8 dB
   Best params:   {
     'optimization.learning_rate': 0.000312,
     'optimization.optimizer_type': 'adamw',
     'model.model_type': 'enhanced_unet',
     'model.model_kwargs.base_channels': 64,
     ...
   }

   optimization.learning_rate              0.412
   model.model_kwargs.base_channels        0.287
   losses_image_l1_weight                  0.163
   model.model_type                        0.089
   optimization.optimizer_type             0.049


Step 4 — Train Best Configuration to Convergence
==================================================

After HPO, apply the best params to a full training run:

.. code-block:: bash

   # Export best config automatically
   python src/main.py hpo-export \
       --config experiments/hpo/hpo_reconstruction.yaml \
       --output experiments/training/tutorial_06_best.yaml

   # Launch full training with best config
   python src/main.py train \
       --config experiments/training/tutorial_06_best.yaml \
       --override "training.max_iterations=100000" \
       --override "checkpoint.save_interval=5000"

Or apply overrides manually from the best trial:

.. code-block:: bash

   python src/main.py train \
       --config experiments/hpo/hpo_reconstruction.yaml \
       --override "optimization.learning_rate=0.000312" \
       --override "model.model_type=enhanced_unet" \
       --override "model.model_kwargs.base_channels=64" \
       --override "training.max_iterations=100000"


Step 5 — Advanced: Multi-Objective HPO
========================================

Optimize for both PSNR (accuracy) and inference time (efficiency):

.. code-block:: yaml

   hpo:
     n_trials: 50
     direction: [maximize, minimize]     # Multi-objective
     objective_metric: [val_psnr, inference_ms_per_slice]
     sampler: nsga2                      # NSGA-II for Pareto front

Results in a Pareto-optimal set of (PSNR, speed) tradeoffs you can
choose from based on deployment constraints.


HPO Sampler Guide
==================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Sampler
     - Best For
     - Notes
   * - ``tpe``
     - Default — continuous params
     - Tree-structured Parzen Estimator; Bayesian
   * - ``cmaes``
     - Continuous params, smooth landscapes
     - Covariance Matrix Adaptation Evolution Strategy
   * - ``nsga2``
     - Multi-objective optimization
     - Pareto-front aware
   * - ``random``
     - Baseline comparison, fully parallel
     - No Bayesian prior
   * - ``grid``
     - Exhaustive small search spaces
     - Scales exponentially with dimensions

HPO Pruner Guide
=================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Pruner
     - Best For
     - Notes
   * - ``median``
     - Default — prune underperforming trials early
     - Stops trial if below median at any checkpoint
   * - ``hyperband``
     - Large-scale sweeps
     - Successive halving with bracket scheduling
   * - ``none``
     - Short training runs (< 5k steps)
     - No pruning


Key Takeaways
=============

1. **Objective metric at partial training** — 5k steps proxy for 100k
2. **Log-uniform LR** — always sample in log space for learning rates
3. **Parallel workers** — each worker loads the same Optuna storage
4. **Dashboard** — ``optuna-dashboard`` gives live visualization
5. **Export best** — use ``hpo-export`` to get a clean YAML
6. **Pruning saves ~60% of compute** — median pruner eliminates bad trials early


See Also
========

- :doc:`../config_schema_reference` — all YAML keys
- :doc:`../strategies_reference` — choose the right training mode
- :doc:`../models_reference` — model types for the search space
- :doc:`../troubleshooting` — OOM during HPO (reduce ``objective_steps``)
