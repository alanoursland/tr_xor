import torch
import torch.nn as nn
import models
import torch.nn.functional as F
from enum import Enum

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from collections import OrderedDict

@experiment("abs1_mse_pivot_control")
def config_abs1_mse_pivot_control() -> ExperimentConfig:
    config = get_experiment_config("abs1_kaiming")
    config.model = nn.Sequential(OrderedDict([
        ('linear1', models.RandomPivotLinear(2, 1, sigma=0.0)),
        ('activation', models.Abs()),
        ('squeeze', models.Squeeze())
    ]))
    nn.init.kaiming_normal_(config.model.linear1.weight, nonlinearity="relu")
    nn.init.zeros_(config.model.linear1.bias)

    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.01)
    config.analysis.parameter_displacement = False
    config.analysis.plot_hyperplanes = False
    config.analysis.plot_epoch_distribution = False
    config.analysis.plot_parameter_displacement = False

    config.description = "Abs1 model. MSE. SGD."
    return config

@experiment("abs1_mse_pivot")
def config_abs1_mse_pivot() -> ExperimentConfig:
    config = get_experiment_config("abs1_mse_pivot_control")
    config.model = nn.Sequential(OrderedDict([
        ('linear1', models.RandomPivotLinear(2, 1, sigma=1.0)),
        ('activation', models.Abs()),
        ('squeeze', models.Squeeze())
    ]))
    linear1 = config.model.linear1
    nn.init.kaiming_normal_(linear1.weight, nonlinearity="relu")
    nn.init.zeros_(linear1.bias)

    config.description = "Abs1 model. MSE. SGD. Random pivot locations."
    return config

@experiment("relu1_mse_pivot_control")
def config_relu1_mse_pivot_control() -> ExperimentConfig:
    config = get_experiment_config("relu1_kaiming")
    config.model = nn.Sequential(OrderedDict([
        ('linear1', models.RandomPivotLinear(2, 2, sigma=0.0)),
        ('activation', nn.ReLU()),
        ('sum', models.Sum(dim=1, keepdim=True)),
        ('squeeze', models.Squeeze()),
    ]))
    with torch.no_grad():
        linear1 = config.model.linear1
        nn.init.kaiming_normal_(linear1.weight, nonlinearity="relu")
        nn.init.zeros_(linear1.bias)
        # mirror initialization
        midpoint = linear1.weight.shape[0] // 2
        linear1.weight[midpoint:] = -linear1.weight[:midpoint].clone()
        linear1.bias[midpoint:] = -linear1.bias[:midpoint].clone()

    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)
    config.training.loss_change_patience=20
    config.training.epochs = 2000
    config.execution.num_runs = 500

    config.analysis.parameter_displacement = False
    config.analysis.plot_hyperplanes = False
    config.analysis.plot_epoch_distribution = False
    config.analysis.plot_parameter_displacement = False
    config.analysis.dead_data_analysis = False

    config.description = "Relu1 model. MSE. SGD."
    return config

@experiment("relu1_mse_pivot")
def config_relu1_mse_pivot() -> ExperimentConfig:
    config = get_experiment_config("relu1_mse_pivot_control")
    config.model = nn.Sequential(OrderedDict([
        ('linear1', models.RandomPivotLinear(2, 2, sigma=1.0)),
        ('activation', nn.ReLU()),
        ('sum', models.Sum(dim=1, keepdim=True)),
        ('squeeze', models.Squeeze()),
    ]))
    with torch.no_grad():
        linear1 = config.model.linear1
        nn.init.kaiming_normal_(linear1.weight, nonlinearity="relu")
        nn.init.zeros_(linear1.bias)
        # mirror initialization
        midpoint = linear1.weight.shape[0] // 2
        linear1.weight[midpoint:] = -linear1.weight[:midpoint].clone()
        linear1.bias[midpoint:] = -linear1.bias[:midpoint].clone()

    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)

    config.description = "Relu1 model. MSE. SGD. Random pivot locations."
    return config

