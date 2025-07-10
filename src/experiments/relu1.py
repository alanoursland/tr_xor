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
    """Factory function for ReLU XOR experiment."""
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
        description="Centered XOR with two nodes, ReLU, sum, and normal init.",
        logging=LoggingConfig(train_epochs=50),
    )


@experiment("relu1_kaiming")
def config_relu1_kaiming() -> ExperimentConfig:
    """Kaiming initialization variant of relu1."""
    config = get_experiment_config("relu1_normal")
    config.model.init_kaiming()
    config.description = "Centered XOR with single absolute value unit and kaiming init."
    return config


@experiment("relu1_reinit")
def config_relu1_reinit() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(config.model.init_normal, config.data.x, 100)
    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized."
    )
    return config


@experiment("relu1_reinit_margin")
def config_relu1_reinit_margin() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(config.model.init_normal, config.data.x, 100, min_threshold=0.3)
    config.execution.num_runs = 500
    config.description = "Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized with margin."
    return config


@experiment("relu1_bhs")
def config_relu1_bhs() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(
        lambda: config.model.init_bounded_hypersphere(config.model.init_normal, radius=1.4),
        config.data.x,
        100,
        min_threshold=0.3,
    )
    config.training.epochs = 2000
    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and bounded hypersphere initialization with norm weights."
    )
    return config


@experiment("relu1_monitor")
def config_relu1_monitor() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment with early-failure monitor."""
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
        ("Centered XOR with two nodes, ReLU, sum, normal init, " "and early-failure degeneracy detection."),
    )
    return config


@experiment("relu1_mirror")
def config_relu1_mirror() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    config.model.init_mirror()
    config.execution.num_runs = 1000
    config.description = "Centered XOR with two nodes, ReLU, sum, and mirrored normal init."
    return config

@experiment("relu1_leaky")
def config_relu1_leaky() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=nn.LeakyReLU()).init_normal()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.description = "Centered XOR with two nodes, Leaky ReLU, sum, and normal init."
    return config

@experiment("relu1_biased")
def config_relu1_biased() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    nn.init.normal_(config.model.linear1.bias, mean=0.1, std=0.01)
    config.description = "Centered XOR with two nodes, ReLU activation, output sum, and normal weight/bias init. "
    return config


@experiment("relu1_elu")
def config_relu1_elu() -> ExperimentConfig:
    """Factory function for ELU XOR experiment."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=torch.nn.ELU())
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "Centered XOR with two nodes, ELU, sum, and normal init."
    return config


@experiment("relu1_prelu")
def config_relu1_prelu() -> ExperimentConfig:
    """Factory function for PReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=torch.nn.PReLU())
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "Centered XOR with two nodes, PReLU, sum, and normal init."
    return config


@experiment("relu1_anneal")
def config_relu1_anneal() -> ExperimentConfig:
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


@experiment("relu1_init_dist")
def config_relu1_init_dist() -> ExperimentConfig:
    config = get_experiment_config("relu1_normal")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.0)
    config.training.loss_change_patience = None
    config.training.epochs = 0
    config.execution.num_runs = 100

    config.analysis.parameter_displacement=False
    config.analysis.distance_to_hyperplanes=False
    config.analysis.hyperplane_clustering=False
    config.analysis.mirror_weight_detection=False
    config.analysis.failure_angle_analysis=False
    config.analysis.dead_data_analysis=False
    config.analysis.plot_hyperplanes=False
    config.analysis.plot_epoch_distribution=False
    config.analysis.plot_parameter_displacement=False
    config.analysis.plot_failure_angles=False

    # config.training.stop_training_loss_threshold = 1e-3
    # config.execution.num_runs = 1
    config.description = "Samples initial loss of relu1_normal. "
    return config

@experiment("relu1_reinit_50th")
def config_relu1_reinit_50th() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    loss_threshold = 4.68e-01
    config = get_experiment_config("relu1_normal")

    x = config.data.x
    y = config.data.y
    loss_fn = config.training.loss_function

    while True:
        config.model.init_normal()
        with torch.no_grad():
            y_pred = config.model(x)
            loss = loss_fn(y_pred, y).item()
        if loss < loss_threshold:
            break

    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and normal init. Data is reinitialized until initial loss is less than 4.68e-01."
    )
    return config

@experiment("relu1_reinit_25th")
def config_relu1_reinit_25th() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    loss_threshold = 3.25e-01
    config = get_experiment_config("relu1_normal")

    x = config.data.x
    y = config.data.y
    loss_fn = config.training.loss_function

    while True:
        config.model.init_normal()
        with torch.no_grad():
            y_pred = config.model(x)
            loss = loss_fn(y_pred, y).item()
        if loss < loss_threshold:
            break

    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and normal init. Data is reinitialized until initial loss is less than 3.25e-01."
    )
    return config

@experiment("relu1_reinit_0th")
def config_relu1_reinit_0th() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    loss_threshold = 7.24e-02
    config = get_experiment_config("relu1_normal")

    x = config.data.x
    y = config.data.y
    loss_fn = config.training.loss_function

    while True:
        config.model.init_normal()
        with torch.no_grad():
            y_pred = config.model(x)
            loss = loss_fn(y_pred, y).item()
        if loss < loss_threshold:
            break

    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and normal init. Data is reinitialized until initial loss is less than 7.24e-02."
    )
    return config

@experiment("relu1_reinit_50th_bad")
def config_relu1_reinit_50th_bad() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    loss_threshold = 4.68e-01
    config = get_experiment_config("relu1_normal")

    x = config.data.x
    y = config.data.y
    loss_fn = config.training.loss_function

    while True:
        config.model.init_normal()
        with torch.no_grad():
            y_pred = config.model(x)
            loss = loss_fn(y_pred, y).item()
        if loss > loss_threshold:
            break

    config.description = (
        "Centered XOR with two nodes, ReLU, sum, and normal init. Data is reinitialized until initial loss is greater than 4.68e-01."
    )
    return config

