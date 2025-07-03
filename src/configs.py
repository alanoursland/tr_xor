# configs.py - Experiment Configuration for Prototype Surface Experiments

"""
Comprehensive experiment configuration system for prototype surface research.
Provides structured experiment definitions, validation, inheritance, and parameter sweep
capabilities. Designed to enable systematic investigation of prototype surface theory across different
model architectures, activation functions, and training configurations.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from data import xor_data_centered, xor_labels_T1, xor_labels_one_hot
import torch
import torch.nn as nn
import models
import monitor
import itertools

# ==============================================================================
# Configuration Schema and Types
# ==============================================================================


# AccuracyFn defines the signature for accuracy calculation functions used in experiments.
#
# Arguments:
#     output (torch.Tensor): The raw output of the model. This is typically the logits
#         from the final layer and may vary in shape depending on the task:
#         - Shape (N, C) for multi-class classification (e.g., BCE with 2 outputs, CE)
#         - Shape (N,) for binary classification with a single output
#
#     target (torch.Tensor): The ground truth labels provided in the dataset. This may be:
#         - A tensor of shape (N,) containing integer class indices (e.g., for CE)
#         - A tensor of shape (N, C) containing one-hot or multi-label float targets (e.g., for BCE)
#
# Returns:
#     float: A scalar accuracy score between 0.0 and 1.0, representing the mean number of
#     correct predictions across the batch.
#
# This abstraction allows each experiment to specify its own logic for interpreting outputs
# and computing correctness, especially useful when using different output encodings (e.g.,
# logits, probabilities, one-hot) or task types (classification, regression, multi-label).
AccuracyFn = Callable[[torch.Tensor, torch.Tensor], float]


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


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""

    optimizer: torch.optim.Optimizer = None
    eps: float = 1e-8

    loss_function: torch.nn.Module = None

    epochs: int = None
    batch_size: int = None

    # Training Monitor
    health_monitor: Optional[Any] = field(default=None)

    # Convergence Detection
    stop_training_loss_threshold: Optional[float] = None

    # Nonconvergence Detection
    early_stopping: bool = False
    patience: int = 50
    min_delta: float = 1e-6
    restore_best_weights: bool = True

    # Convergence criteria
    loss_change_threshold: Optional[float] = None
    loss_change_patience: int = 10

    # Gradient clipping
    gradient_clipping: bool = False
    max_grad_norm: float = 1.0

    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if self.loss_function is not None:
            del self.loss_function
            self.loss_function = None


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""

    x: torch.Tensor
    y: torch.Tensor

    # Optional metadata
    description: str = ""


@dataclass
class AnalysisConfig:
   # Core analyses (always run - these are cheap and fundamental)
   accuracy_fn: Optional[AccuracyFn] = None
   accuracy_threshold: float = 1.0  # for filtering successful runs
   
   # Individual analyses
   weight_clustering: bool = True           # DBSCAN clustering of final weights
   parameter_displacement: bool = False     # Initial→final weight angle/norm analysis
   distance_to_hyperplanes: bool = True     # Clusters linear layer distances to classes
   hyperplane_clustering: bool = False      # Clusters hyperplane positions
   mirror_weight_detection: bool = False    # Detect w_i ≈ -w_j pairs (ReLU-specific)
   failure_angle_analysis: bool = False     # Initial angle analysis for failed runs
   dead_data_analysis: bool = False         # ReLU-specific dead data detection
   
   # Individual visualizations
   plot_hyperplanes: bool = False            # Geometric plots per run
   plot_epoch_distribution: bool = False     # Convergence timing histogram
   plot_parameter_displacement: bool = False # Angle/magnitude vs epochs
   plot_failure_angles: bool = False         # Success vs failure angle histogram
   
   # Visualization settings
   plot_format: str = "png"
   plot_dpi: int = 300
   
   # Analysis parameters
   analysis_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(-2.5, 2.5), (-2.5, 2.5)])
   analysis_resolution: int = 100

@dataclass
class ExecutionConfig:
    """Configuration for experiment execution."""

    num_runs: int = 10
    random_seeds: Optional[List[int]] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    skip_existing: bool = True  # Don't rerun existing experiments
    normalize_weights: bool = False # Normalize weights before saving

    # Parallel execution
    parallel_runs: bool = False
    max_workers: Optional[int] = None

    # Output and logging
    output_dir: str = "results"
    experiment_name: str = "unnamed_experiment"
    save_intermediate: bool = False
    save_frequency: int = 100  # epochs

    # Logging
    logging_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    detailed_logging: bool = False

    # Resource management
    memory_limit: Optional[int] = None  # MB
    time_limit: Optional[int] = None  # seconds

    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False  # For CUDA reproducibility vs performance


@dataclass
class LoggingConfig:
    """Configuration for logging information."""

    train_epochs: int = 1000  # number of training epochs between logging


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    model: torch.nn.Module
    training: TrainingConfig
    data: DataConfig
    analysis: AnalysisConfig
    execution: ExecutionConfig
    logging: LoggingConfig

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_by: str = "prototype_framework"
    notes: str = ""

    # Dependencies and inheritance
    base_config: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Clean up all config resources."""
        if self.model is not None:
            # Move model to CPU first, then delete
            self.model = self.model.cpu()
            del self.model
            self.model = None

        if self.training is not None:
            self.training.cleanup()

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# Main Experiments Registry
# ==============================================================================


def get_experiment_config(name: str) -> ExperimentConfig:
    """
    Retrieve experiment configuration by name.

    Args:
        name: Name of experiment configuration

    Returns:
        Complete experiment configuration
    """
    if name not in experiments:
        available = list(experiments.keys())
        raise KeyError(f"Unknown experiment '{name}'. Available experiments: {available}")

    # Call the factory function to create the config
    config_factory = experiments[name]
    config = config_factory()

    return config


def list_experiments(category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[str]:
    """
    List available experiment configurations.

    Args:
        category: Filter by experiment category
        tags: Filter by experiment tags

    Returns:
        List of experiment names matching criteria
    """
    # Start with all experiment names
    experiment_names = list(experiments.keys())

    # If no filtering requested, return all
    if category is None and tags is None:
        return sorted(experiment_names)

    # Filter by category and/or tags if requested
    filtered_names = []

    for name in experiment_names:
        # Create config to check its metadata
        try:
            config = experiments[name]()

            # Check category filter
            if category is not None:
                # Since we simplified, we can infer category from name or description
                # Or you could add a category field to ExperimentConfig
                if category.lower() not in name.lower() and category.lower() not in config.description.lower():
                    continue

            # Check tags filter
            if tags is not None:
                # Would need to add tags field to ExperimentConfig for this to work
                # For now, skip tag filtering
                pass

            filtered_names.append(name)

        except Exception:
            # If config creation fails, skip this experiment
            continue

    return sorted(filtered_names)


experiments: Dict[str, Callable[[], ExperimentConfig]] = {}


def experiment(name: str):
    """Decorator to register and label experiment configuration functions."""

    def decorator(func: Callable[[], ExperimentConfig]):
        def wrapped_func():
            config = func()
            config.execution.experiment_name = name  # Inject decorator name
            return config

        experiments[name] = wrapped_func
        return wrapped_func

    return decorator


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


@experiment("relu1_reinit")
def config_relu1_reinit() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    config = get_experiment_config("relu1_normal")
    config.model.reinit_dead_data(config.model.init_normal, config.data.x, 100)
    config.description = "Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized."
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
        config.data.x, 100, min_threshold=0.3
    )
    config.training.epochs = 2000
    config.description = "Centered XOR with two nodes, ReLU, sum, and bounded hypersphere initialization with norm weights."
    return config

@experiment("relu1_monitor")
def config_relu1_monitor() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment with early-failure monitor."""
    config = get_experiment_config("relu1_normal")
    dataset_size = config.data.x.shape[0]  # == 4
    hook_manager = monitor.SharedHookManager(config.model)
    health_monitor = monitor.CompositeMonitor(
        [
            monitor.DeadSampleMonitor(hook_manager, dataset_size=dataset_size, patience=5, classifier_threshold=0.5),
            monitor.BoundsMonitor(hook_manager, dataset_size=dataset_size, radius=1.5),
        ]
    )
    config.training.health_monitor = health_monitor
    config.description=("Centered XOR with two nodes, ReLU, sum, normal init, " "and early-failure degeneracy detection."),
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
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
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
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    config.description = "Centered XOR with 2-output MSE loss, L2 reg, using a single Abs unit."
    return config


@experiment("abs2_single_bce_confidence")
def config_abs2_single_bce_confidence() -> ExperimentConfig:
    config = get_experiment_config("abs2_single_bce")
    config.model = models.Model_Xor2_Confidence(middle=1, activation=models.Abs()).init()
    # Recreate optimizer with new model
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99))
    config.description = "Centered XOR with 2-output BCE loss using a single Abs unit. Includes a confidence final layer."
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
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
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
    config.training.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
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
