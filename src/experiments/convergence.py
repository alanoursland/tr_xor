import torch
import torch.nn as nn
import models
import monitor
import itertools
import torch.nn.functional as F
from enum import Enum

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_T1, xor_labels_one_hot

@experiment("abs1_convergence")
def config_abs1_convergence() -> ExperimentConfig:
    config = get_experiment_config("abs1_normal")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)
    config.training.loss_change_patience = None

    # Add parameter trace monitor
    trace_monitor = monitor.ParameterTraceMonitor(
        config=config,
        dataset_size=config.data.x.shape[0],  # Should be 4 for XOR data
        save_frequency=1,  # Save every epoch for detailed convergence analysis
    )
    # print(trace_monitor.output_dir)
    # print(trace_monitor.experiment_name)
    # print(trace_monitor.trace_subdir)
    # print(trace_monitor.save_frequency)
    config.training.training_monitor = trace_monitor
    config.execution.num_runs = 100
    config.description = (
        "Abs1 experiment tailored to study convergence speed between init and final with parameter tracing."
    )
    config.logging.train_epochs = 10
    return config

@experiment("relu1_convergence")
def config_relu1_convergence() -> ExperimentConfig:
    config = get_experiment_config("relu1_normal")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)
    config.training.loss_change_patience = 20

    # Add parameter trace monitor
    trace_monitor = monitor.ParameterTraceMonitor(
        config=config,
        dataset_size=config.data.x.shape[0],  # Should be 4 for XOR data
        save_frequency=1,  # Save every epoch for detailed convergence analysis
    )
    config.execution.random_seeds = [501]
    config.training.training_monitor = trace_monitor
    config.training.epochs = 600
    config.execution.num_runs = 500 # 54% get 100% and we want at least 100 good runs
    config.description = (
        "Relu1 experiment tailored to study convergence speed between init and final with parameter tracing."
    )
    config.logging.train_epochs = 10
    return config

@experiment("relu2_convergence")
def config_relu2_convergence() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse_sgd")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)
    config.training.loss_change_patience = 20

    # Add parameter trace monitor
    trace_monitor = monitor.ParameterTraceMonitor(
        config=config,
        dataset_size=config.data.x.shape[0],  # Should be 4 for XOR data
        save_frequency=1,  # Save every epoch for detailed convergence analysis
    )
    config.training.training_monitor = trace_monitor
    config.training.epochs = 600
    config.execution.num_runs = 500 # 26.8% get 100% and we want at least 100 good runs
    config.description = (
        "Relu2 experiment tailored to study convergence speed between init and final with parameter tracing."
    )
    config.logging.train_epochs = 10
    return config

@experiment("relu1_init_dist")
def config_relu1_init_dist() -> ExperimentConfig:
    """No training; just log initial loss distribution."""
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
    config.description = "Init-loss scan."
    return config

@experiment("relu1_reinit_50th")
def config_relu1_reinit_50th() -> ExperimentConfig:
    """Reinit until initial loss ≤ 50 th percentile."""
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
        "Reinit ≤ 50 % loss."
    )
    return config

@experiment("relu1_reinit_25th")
def config_relu1_reinit_25th() -> ExperimentConfig:
    """Reinit until initial loss ≤ 25 th percentile."""
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
        "Reinit ≤ 25 % loss."
    )
    return config

@experiment("relu1_reinit_0th")
def config_relu1_reinit_0th() -> ExperimentConfig:
    """Reinit until initial loss ≤ best observed."""
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
        "Reinit best-loss."
    )
    return config

@experiment("relu1_reinit_50th_bad")
def config_relu1_reinit_50th_bad() -> ExperimentConfig:
    """Reinit until initial loss ≥ 50 th percentile (worst half)."""
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
        "Reinit ≥ 50 % loss."
    )
    return config

