"""
abs1.py — Absolute-Value XOR Experiment Suite
─────────────────────────────────────────────

Purpose
-------
This file contains the **minimal set of experiments** used to validate and
illustrate the “prototype-surface” theory outlined in *core_theory.md*.  In
particular, it serves four research objectives:

1. **Prototype-surface demonstration**  
   Show that a single Linear + |·| unit can learn a hyperplane that *exactly*
   matches the XOR geometry.  The learned surface intersects the class-0 points
   and sits √2 units from the class-1 points, confirming the analytic solution.

2. **Learning-dynamics study**  
   Measure how different weight-initialisation schemes (normal, tiny, large,
   Xavier, Kaiming) affect convergence speed, weight rotation, and distance-
   field formation in this minimal setting.

3. **Optimizer comparison**  
   Contrast Adam (lr = 0.01) with vanilla SGD under several constant learning
   rates.  We show that SGD with η ≈ 0.5 attains one-to-two-epoch convergence,
   matching the Newton step derived for this quadratic loss.

4. **Didactic baseline**  
   Provide a reproducible, human-readable reference implementation of prototype
  -surface learning on the smallest non-linearly-separable Boolean task.

Why XOR?
--------
XOR is the smallest “hard” Boolean problem—it cannot be solved by a single
linear threshold unit—yet it is rich enough to force the model to carve two
disjoint regions.  This makes it an ideal sandbox for observing how an |·|
activation positions its hyperplane and how different inits/optimizers steer
learning dynamics.

Shared Experimental Skeleton
----------------------------
* **Model** : Input(2) → Linear(2 → 1) → Abs() → Squeeze → Output(1)  
    (analytically : y = | w·x + b |)
* **Loss**   : Mean-squared error on binary targets
* **Dataset**: Centered XOR {(-1,-1),( 1, 1) → 0; (-1, 1),( 1,-1) → 1}
* **Early stop** : loss < 1 × 10⁻⁷ (max 1000 epochs unless noted)
* **Initialisers** : normal(0,0.5), tiny, large, Xavier, Kaiming
* **Optimizers**  :  
    – Adam (lr = 0.01, β = 0.9/0.99)   *default*  
    – SGD (lr ∈ {0.1, 0.4–0.9})        *`abs1_*_mse` variants*

Each experiment is registered with `@experiment(...)`; all hyper-parameters
except the chosen initialiser/optimizer are inherited from the `abs1_normal`
baseline.

The generated analysis artifacts (`analysis_abs1_<variant>.md`) report epochs-
to-convergence, hyperplane clustering, weight-norm trajectories, etc.  Those
results are *not* embedded here to keep the source concise; this header gives
future readers enough context to reproduce and interpret the experiments.
"""

import torch
import torch.nn as nn
import monitor
import itertools
import torch.nn.functional as F
import models
from enum import Enum

from models import init_model, with_normal_weights, with_kaiming_weights, with_xavier_weights, with_tiny_weights, with_large_weights, with_zero_bias
from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_T1, accuracy_binary_threshold
from collections import OrderedDict

def create_abs1(activation=models.Abs()):
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(2,1)),
        ('activation', activation),
        ('squeeze', models.Squeeze())
    ]))

@experiment("abs1_normal")
def config_abs1_normal() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = init_model(create_abs1(), with_normal_weights, with_zero_bias)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(
            optimizer=optimizer, loss_function=loss_function, epochs=1000, stop_training_loss_threshold=1e-7
        ),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(
            accuracy_fn=accuracy_binary_threshold,
            # Core analyses (default enabled)
            parameter_displacement=True,
            distance_to_hyperplanes=True,
            hyperplane_clustering=True,
            # Specialized analyses (default disabled)
            mirror_weight_detection=False,
            failure_angle_analysis=False,
            dead_data_analysis=False,
            # Visualizations (default disabled for speed)
            plot_hyperplanes=True,
            plot_epoch_distribution=True,
            plot_parameter_displacement=True,
            plot_failure_angles=False,
        ),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Abs1 experiment with normal weight initialization (std=0.5). Explores convergence from standard Gaussian init.",
        logging=LoggingConfig(train_epochs=200),
    )

@experiment("abs1_tiny")
def config_abs1_tiny() -> ExperimentConfig:
    """Tiny initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_tiny_weights, with_zero_bias)
    config.description="Abs1 experiment with tiny normal init (std=0.1). Tests impact of initialization on learning."
    return config


@experiment("abs1_large")
def config_abs1_large() -> ExperimentConfig:
    """Large initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_large_weights, with_zero_bias)
    config.training.epochs = 2000  # Slower convergence expected
    config.description="Abs1 experiment with tiny normal init (std=4.0). Tests impact of initialization on learning."
    return config


@experiment("abs1_kaiming")
def config_abs1_kaiming() -> ExperimentConfig:
    """Kaiming initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_kaiming_weights, with_zero_bias)
    config.description="Abs1 experiment with Kaiming weight initialization. Tests impact of initialization on learning."
    return config


@experiment("abs1_xavier")
def config_abs1_xavier() -> ExperimentConfig:
    """Xavier initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_xavier_weights, with_zero_bias)
    config.description="Abs1 experiment with Kaiming weight initialization. Tests impact of initialization on learning."
    return config

@experiment("abs1_kaiming_mse")
def config_abs1_kaiming_mse() -> ExperimentConfig:
    """Kaiming initialization variant of abs1."""
    config = get_experiment_config("abs1_kaiming")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.5)
    config.description="Abs1 experiment with Kaiming weight initialization and SGD. Tests impact of optimzer on learning."
    return config


