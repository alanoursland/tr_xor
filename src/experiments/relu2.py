import torch
import torch.nn as nn
import models
import torch.nn.functional as F

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_one_hot, accuracy_one_hot

@experiment("relu2_two_bce")
def config_relu2_two_bce() -> ExperimentConfig:
    model = models.Model_Xor2(middle=2, activation=nn.ReLU()).init()
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
            parameter_displacement=False,
            distance_to_hyperplanes=True,
            hyperplane_clustering=True,
            # ReLU-specific analyses
            mirror_weight_detection=True,
            failure_angle_analysis=False,
            dead_data_analysis=False,
            # Visualizations (off for these experiments)
            plot_hyperplanes=False,
            plot_epoch_distribution=False,
            plot_parameter_displacement=False,
            plot_failure_angles=False,
        ),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss using two ReLU units.",
        logging=LoggingConfig(train_epochs=50),
    )


@experiment("relu2_two_bce_l2reg")
def config_relu2_two_bce_l2reg() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_bce")
    config.training.optimizer = torch.optim.Adam(
        config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1
    )
    config.description = "Centered XOR with 2-output BCE loss, L2 reg, and two ReLU units."
    return config


@experiment("relu2_two_mse")
def config_relu2_two_mse() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_bce")
    config.training.loss_function = nn.MSELoss()
    config.description = "Centered XOR with 2-output MSE loss using two ReLU units."
    return config


@experiment("relu2_two_mse_l2reg")
def config_relu2_two_mse_l2reg() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.training.optimizer = torch.optim.Adam(
        config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1
    )
    config.description = "Centered XOR with 2-output MSE loss, L2 reg, and two ReLU units."
    return config


@experiment("relu2_one_mse_norm")
def config_relu2_one_mse_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse_norm")
    config.model = models.Model_Xor2(middle=1, activation=nn.ReLU()).init()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.accuracy_threshold = 0.75
    config.analysis.plot_hyperplanes = True
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss and one ReLU unit. Weight normalization post training."
    return config


@experiment("relu2_three_mse_norm")
def config_relu2_three_bce_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.model = models.Model_Xor2(middle=3, activation=nn.ReLU()).init()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss and three ReLU units. Weight normalization post training."
    return config


@experiment("relu2_four_mse_norm")
def config_relu2_four_bce_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.model = models.Model_Xor2(middle=4, activation=nn.ReLU()).init()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss and four ReLU units. Weight normalization post training."
    return config


@experiment("relu2_eight_mse_norm")
def config_relu2_eight_bce_norm() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.model = models.Model_Xor2(middle=8, activation=nn.ReLU()).init()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.execution.normalize_weights = True
    config.description = "Centered XOR with 2-output BCE loss and eight ReLU units. Weight normalization post training."
    return config

@experiment("relu2_two_bce_eater")
def config_relu2_two_bce_eater() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_bce")
    config.model = models.Model_Xor2_Eater(middle=1, activation=nn.ReLU(), max_points=4).init()
    # Recreate optimizer with new model
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.description = "Centered XOR with 2-output BCE loss using two ReLU units. Includes 'eater' layers intended to regularize linear layers."
    return config


@experiment("relu2_two_mse_eater")
def config_relu2_two_mse_eater() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.model = models.Model_Xor2_Eater(middle=1, activation=nn.ReLU(), max_points=4).init()
    config.training.loss_function = nn.MSELoss()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.accuracy_threshold = 0.75
    config.analysis.parameter_displacement = False
    config.description = "Centered XOR with 2-output MSE loss using two ReLU units. Includes 'eater' layers intended to regularize linear layers."
    return config

@experiment("relu2_init_dist")
def config_relu2_init_dist() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
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
    config.description = "Samples initial loss of relu2_two_mse. "
    return config

@experiment("relu2_two_mse_sgd")
def config_relu2_two_mse_sgd() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.1)
    config.training.epochs = 600
    config.training.loss_change_patience = None

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

    config.description = "relu2_two_mse with SGD optimizer."
    return config

@experiment("relu2_reinit_0th")
def config_relu1_reinit_0th() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    loss_threshold = 3.26e-01
    config = get_experiment_config("relu2_two_mse_sgd")

    x = config.data.x
    y = config.data.y
    loss_fn = config.training.loss_function

    while True:
        config.model.init()
        with torch.no_grad():
            y_pred = config.model(x)
            loss = loss_fn(y_pred, y).item()
        if loss < loss_threshold:
            break

    config.training.loss_change_patience = None

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

    config.descrption = f"Centered XOR with 2-output MSE loss using two ReLU units. Initial loss < {loss_threshold}"
    return config

