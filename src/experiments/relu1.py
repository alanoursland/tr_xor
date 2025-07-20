import torch
import torch.nn as nn
import models
import monitor
import itertools
import torch.nn.functional as F
from enum import Enum

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_T1, accuracy_binary_threshold

@experiment("relu1_normal")
def config_relu1_normal() -> ExperimentConfig:
    """Baseline: normal init, 2-ReLU sum XOR."""
    model = models.Model_ReLU1().init_normal()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=800,
            stop_training_loss_threshold=1e-7,
            loss_change_threshold=1e-24,
            loss_change_patience=10,
        ),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(
            accuracy_fn=accuracy_binary_threshold,
            # Core analyses (default enabled)
            parameter_displacement=True,
            distance_to_hyperplanes=True,
            hyperplane_clustering=True,
            # Specialized analyses (default disabled)
            mirror_weight_detection=True,
            failure_angle_analysis=True,
            dead_data_analysis=True,
            # Visualizations (default disabled for speed)
            plot_hyperplanes=False,
            plot_epoch_distribution=True,
            plot_parameter_displacement=True,
            plot_failure_angles=True,
        ),
        execution=ExecutionConfig(num_runs=50, skip_existing=False, random_seeds=[18]),
        description="Baseline ReLU XOR.",
        logging=LoggingConfig(train_epochs=50),
    )


@experiment("relu1_kaiming")
def config_relu1_kaiming() -> ExperimentConfig:
    """Kaiming (He) weight initialization."""
    config = get_experiment_config("relu1_normal")
    config.model.init_kaiming()
    config.description = "Kaiming init."
    return config


@experiment("relu1_reinit")
def config_relu1_reinit() -> ExperimentConfig:
    """Auto-restart if any sample stays dead."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(config.model.init_normal, config.data.x, 100)
    config.description = (
        "Auto-reinit on dead data."
    )
    return config


@experiment("relu1_reinit_margin")
def config_relu1_reinit_margin() -> ExperimentConfig:
    """Reinit on dead data with 0.3 margin."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(config.model.init_normal, config.data.x, 100, min_threshold=0.3)
    config.execution.num_runs = 500
    config.description = "Reinit on margin 0.3."
    return config


@experiment("relu1_bhs")
def config_relu1_bhs() -> ExperimentConfig:
    """Tangent-to-hypersphere init; all samples active."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(
        lambda: config.model.init_bounded_hypersphere(config.model.init_normal, radius=1.4),
        config.data.x,
        100,
        min_threshold=0.3,
    )
    config.training.epochs = 2000
    config.description = (
        "Bounded-sphere init."
    )
    return config


@experiment("relu1_monitor")
def config_relu1_monitor() -> ExperimentConfig:
    """Runtime monitors fix dead samples / large norms"""
    config = get_experiment_config("relu1_normal")
    dataset_size = config.data.x.shape[0]  # == 4
    hook_manager = monitor.SharedHookManager(config.model)
    training_monitor = monitor.CompositeMonitor(
        [
            monitor.DeadSampleMonitor(hook_manager, dataset_size=dataset_size, patience=5, classifier_threshold=0.5),
            monitor.BoundsMonitor(hook_manager, dataset_size=dataset_size, radius=1.5),
        ]
    )
    config.training.training_monitor = training_monitor
    config.description = (
        ("Runtime monitors."),
    )
    return config


@experiment("relu1_mirror")
def config_relu1_mirror() -> ExperimentConfig:
    """Mirror-symmetric weight vectors at start."""
    config = get_experiment_config("relu1_normal")
    config.model.init_mirror()
    config.execution.num_runs = 1000
    config.description = "Mirrored init."
    return config

@experiment("relu1_leaky_1e-2")
def config_relu1_leaky_1En2() -> ExperimentConfig:
    """Standard LeakyReLU, slope 0.01."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=nn.LeakyReLU()).init_normal()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.description = "LeakyReLU 0.01. "
    return config

@experiment("relu1_leaky_-1e-2")
def config_relu1_leaky_n1En2() -> ExperimentConfig:
    """LeakyAbs: symmetric negative leak (≈|z|)."""
    config = get_experiment_config("relu1_normal")

    models.Model_ReLU1(activation=models.LeakyAbs()).init_normal()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "LeakyAbs 0.01."
    return config

@experiment("relu1_biased")
def config_relu1_biased() -> ExperimentConfig:
    """Positive hidden bias (μ = 0.1) to keep units alive."""
    config = get_experiment_config("relu1_normal")
    nn.init.normal_(config.model.linear1.bias, mean=0.1, std=0.01)
    config.description = "Biased ReLU init."
    return config


@experiment("relu1_elu")
def config_relu1_elu() -> ExperimentConfig:
    """ELU activation replaces ReLU."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=torch.nn.ELU())
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "ELU activation."
    return config


@experiment("relu1_prelu")
def config_relu1_prelu() -> ExperimentConfig:
    """Trainable PReLU activation."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=torch.nn.PReLU())
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "PReLU activation."
    return config


@experiment("relu1_anneal")
def config_relu1_anneal() -> ExperimentConfig:
    """Adds noise when entropy high (annealing rescue)."""
    config = get_experiment_config("relu1_normal")

    hook_manager = monitor.SharedHookManager(config.model)
    config.training.training_monitor = monitor.AnnealingMonitor(
        hook_manager=hook_manager,
        dataset_size=4,
        loss_fn=config.training.loss_function,
        base_noise_level=0.1,
        annealing_threshold=0.1,
    )
    config.training.loss_change_patience = None
    config.training.epochs = 5000
    # config.training.stop_training_loss_threshold = 1e-3
    # config.execution.num_runs = 1
    config.description = "Centered XOR with two nodes, ReLU activation, output sum, and error driven annealing. "
    return config

