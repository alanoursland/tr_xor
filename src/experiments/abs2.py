import torch
import torch.nn as nn
import models
import monitor
import itertools
import torch.nn.functional as F
from enum import Enum

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_one_hot, accuracy_one_hot

@experiment("abs2_single_bce")
def config_abs2_single_bce() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=5000,
            stop_training_loss_threshold=1e-7,
            loss_change_threshold=1e-24,
            loss_change_patience=10,
        ),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(
            accuracy_fn=accuracy_one_hot,
            # Core analyses
            weight_clustering=True,
            parameter_displacement=False,
            distance_to_hyperplanes=True,
            hyperplane_clustering=True,
            # Specialized analyses
            mirror_weight_detection=False,
            failure_angle_analysis=False,
            dead_data_analysis=False,
            # Visualizations
            plot_hyperplanes=False,
            plot_epoch_distribution=False,
            plot_parameter_displacement=False,
            plot_failure_angles=False,
        ),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50),
    )


@experiment("abs2_single_bce_l2reg")
def config_abs2_single_bce_l2reg() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    # Recreate optimizer with weight decay
    config.training.optimizer = torch.optim.Adam(
        config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1
    )
    config.description = "Centered XOR with 2-output BCE loss, L2 reg, using a single Abs unit."
    return config


@experiment("abs2_single_mse")
def config_abs2_single_mse() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.training.loss_function = nn.MSELoss()
    config.description = "Centered XOR with 2-output MSE loss using a single Abs unit."
    return config


@experiment("abs2_single_mse_l2reg")
def config_abs2_single_mse_l2reg() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_mse")
    # Recreate optimizer with weight decay
    config.training.optimizer = torch.optim.Adam(
        config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1
    )
    config.description = "Centered XOR with 2-output MSE loss, L2 reg, using a single Abs unit."
    return config


@experiment("abs2_single_bce_confidence")
def config_abs2_single_bce_confidence() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.model = models.Model_Xor2_Confidence(middle=1, activation=models.Abs()).init()
    # Recreate optimizer with new model
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.description = (
        "Centered XOR with 2-output BCE loss using a single Abs unit. Includes a confidence final layer."
    )
    return config


@experiment("abs2_single_bce_confidence_l2reg")
def config_abs2_single_bce_confidence_l2reg() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce_confidence")

    # Create grouped parameters for selective L2 regularization
    optimizer_grouped_parameters = [
        {
            "params": itertools.chain(config.model.linear1.parameters(), config.model.linear2.parameters()),
            "weight_decay": 1e-1,  # Apply L2 regularization here
        },
        {"params": config.model.confidence.parameters(), "weight_decay": 0.0},  # Do NOT apply L2 regularization here
    ]
    config.training.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.01, betas=(0.9, 0.99))
    config.description = "Centered XOR with 2-output BCE loss, L2 reg, using a single Abs unit."
    return config

@experiment("abs2_single_bce_norm")
def config_abs2_single_bce_norm() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss using a single Abs unit with manual weight normalization."
    return config


@experiment("abs2_single_mse_norm")
def config_abs2_single_mse_norm() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_mse")
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output MSE loss using a single Abs unit with manual weight normalization."
    return config


@experiment("relu2_two_bce_norm")
def config_relu2_two_bce_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_bce")
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss using two ReLU units with weight normalization."
    return config


@experiment("relu2_two_mse_norm")
def config_relu2_two_mse_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output MSE loss using two ReLU units with weight normalization."
    return config


@experiment("abs2_single_bce_eater")
def config_abs2_single_bce_eater() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.model = models.Model_Xor2_Eater(middle=1, activation=models.Abs(), max_points=4).init()
    # Recreate optimizer with new model
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.description = "Centered XOR with 2-output BCE loss using a single Abs unit. Includes 'eater' layers intended to regularize linear layers."
    return config


@experiment("abs2_single_mse_eater")
def config_abs2_single_mse() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.model = models.Model_Xor2_Eater(middle=1, activation=models.Abs(), max_points=4).init()
    config.training.loss_function = nn.MSELoss()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.description = "Centered XOR with 2-output MSE loss using a single Abs unit. Includes 'eater' layers intended to regularize linear layers."
    return config



