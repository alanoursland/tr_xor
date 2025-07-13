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
        description="Centered XOR with single absolute value unit and normal init.",
        logging=LoggingConfig(train_epochs=200),
    )

@experiment("abs1_tiny")
def config_abs1_tiny() -> ExperimentConfig:
    """Tiny initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_tiny_weights, with_zero_bias)
    config.description = "Centered XOR with single absolute value unit and tiny normal init."
    return config


@experiment("abs1_large")
def config_abs1_large() -> ExperimentConfig:
    """Large initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_large_weights, with_zero_bias)
    config.training.epochs = 2000  # Slower convergence expected
    config.description = "Centered XOR with single absolute value unit and large normal init."
    return config


@experiment("abs1_kaiming")
def config_abs1_kaiming() -> ExperimentConfig:
    """Kaiming initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_kaiming_weights, with_zero_bias)
    config.description = "Centered XOR with single absolute value unit and kaiming init."
    return config


@experiment("abs1_xavier")
def config_abs1_xavier() -> ExperimentConfig:
    """Xavier initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    init_model(config.model, with_xavier_weights, with_zero_bias)
    config.description = "Centered XOR with single absolute value unit and xavier init."
    return config


