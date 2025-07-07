import torch
import torch.nn as nn
import models
import monitor
import itertools
import torch.nn.functional as F
from enum import Enum

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_T1, xor_labels_one_hot

def accuracy_binary_threshold(output: torch.Tensor, target: torch.Tensor) -> float:
    # Squeeze output in case shape is (N, 1)
    output = output.squeeze()

    # Apply threshold at 0.5
    preds = (output >= 0.5).float()

    # Compute accuracy
    return (preds == target).float().mean().item()


def accuracy_one_hot(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(output, dim=1)
    true = torch.argmax(target, dim=1)
    return (preds == true).float().mean().item()


@experiment("abs1_normal")
def config_abs1_normal() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = models.Model_Abs1().init_normal()
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
            weight_clustering=True,
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
    config.model.init_tiny()
    config.description = "Centered XOR with single absolute value unit and tiny normal init."
    return config


@experiment("abs1_large")
def config_abs1_large() -> ExperimentConfig:
    """Large initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    config.model.init_large()
    config.training.epochs = 2000  # Slower convergence expected
    config.description = "Centered XOR with single absolute value unit and large normal init."
    return config


@experiment("abs1_kaiming")
def config_abs1_kaiming() -> ExperimentConfig:
    """Kaiming initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    config.model.init_kaiming()
    config.description = "Centered XOR with single absolute value unit and kaiming init."
    return config


@experiment("abs1_xavier")
def config_abs1_xavier() -> ExperimentConfig:
    """Xavier initialization variant of abs1."""
    config = get_experiment_config("abs1_normal")
    config.model.init_xavier()
    config.description = "Centered XOR with single absolute value unit and xavier init."
    return config


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
            weight_clustering=True,
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
            weight_clustering=True,
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


@experiment("relu1_leaky")
def config_relu1_leaky() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=nn.LeakyReLU()).init_normal()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.analysis.plot_hyperplanes = True
    config.description = "Centered XOR with two nodes, Leaky ReLU, sum, and normal init."
    return config


@experiment("abs1_leaky")
def config_abs1_leaky() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")

    config.model = models.Model_ReLU1(activation=models.LeakyAbs()).init_normal()
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.training.epochs = 5000
    config.analysis.plot_hyperplanes = True
    config.description = "Centered XOR with two nodes, Leaky Abs, sum, and normal init."
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

@experiment("relu1_init_dist")
def config_relu1_init_dist() -> ExperimentConfig:
    config = get_experiment_config("relu1_normal")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.0)
    config.training.loss_change_patience = None
    config.training.epochs = 0
    config.execution.num_runs = 100

    config.analysis.weight_clustering=False
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

@experiment("relu2_init_dist")
def config_relu2_init_dist() -> ExperimentConfig:
    config = get_experiment_config("relu2_two_mse")
    config.training.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.0)
    config.training.loss_change_patience = None
    config.training.epochs = 0
    config.execution.num_runs = 100

    config.analysis.weight_clustering=False
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

    config.analysis.weight_clustering=False
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

    config.analysis.weight_clustering=False
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
