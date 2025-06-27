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
    """Configuration for post-training analysis."""

    # Evaluation functions
    accuracy_fn: Optional[AccuracyFn] = None

    # Geometric analysis
    geometric_analysis: bool = True
    hyperplane_plots: bool = True
    prototype_region_analysis: bool = True
    decision_boundary_analysis: bool = True
    distance_field_analysis: bool = True

    # Weight analysis
    weight_analysis: bool = True
    mirror_pair_detection: bool = False
    weight_evolution_tracking: bool = False
    weight_clustering: bool = False
    symmetry_analysis: bool = True
    failure_angles: bool = False

    # Activation analysis
    activation_analysis: bool = True
    activation_patterns: bool = True
    sparsity_analysis: bool = True
    zero_activation_regions: bool = True
    dead_data_analysis: bool = False

    # Convergence analysis
    convergence_analysis: bool = True
    training_dynamics: bool = True
    loss_landscape_analysis: bool = False

    # Comparative analysis
    cross_run_comparison: bool = True
    statistical_analysis: bool = True
    stability_analysis: bool = True

    # Visualization options
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    plot_dpi: int = 300
    interactive_plots: bool = False
    plot_style: str = "default"

    # Analysis bounds and resolution
    analysis_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(-2.5, 2.5), (-2.5, 2.5)])
    analysis_resolution: int = 100

    # Prototype surface specific analysis
    prototype_surface_analysis: bool = True
    separation_order_analysis: bool = True
    minsky_papert_metrics: bool = True


@dataclass
class ExecutionConfig:
    """Configuration for experiment execution."""

    num_runs: int = 10
    random_seeds: Optional[List[int]] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    skip_existing: bool = True  # Don't rerun existing experiments

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
    train_epochs: int = 1000 # number of training epochs between logging

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
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=400, stop_training_loss_threshold=1e-7),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with single absolute value unit and normal init.",
        logging=LoggingConfig(train_epochs=200)
    )

@experiment("abs1_tiny")
def config_abs1_normal() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = models.Model_Abs1().init_tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=200, stop_training_loss_threshold=1e-7),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with single absolute value unit and tiny normal init.",
        logging=LoggingConfig(train_epochs=200)
    )

@experiment("abs1_large")
def config_abs1_normal() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = models.Model_Abs1().init_large()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=2000, stop_training_loss_threshold=1e-7),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with single absolute value unit and large normal init.",
        logging=LoggingConfig(train_epochs=200)
    )

@experiment("abs1_kaiming")
def config_abs1_kaiming() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = models.Model_Abs1().init_kaiming()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=1000, stop_training_loss_threshold=1e-7),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with single absolute value unit and kaiming init.",
        logging=LoggingConfig(train_epochs=200)
    )

@experiment("abs1_xavier")
def config_abs1_xavier() -> ExperimentConfig:
    """Factory function for absolute value XOR experiment."""
    model = models.Model_Abs1().init_xavier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=1000, stop_training_loss_threshold=1e-7),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with single absolute value unit xavier init.",
        logging=LoggingConfig(train_epochs=200)
    )

@experiment("relu1_normal")
def config_relu1_normal() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    model = models.Model_ReLU1().init_normal()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=800, 
                                stop_training_loss_threshold=1e-7,
                                loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold, convergence_analysis=False, save_plots=True, dead_data_analysis=True, mirror_pair_detection=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False, random_seeds=[18]),
        description="Centered XOR with two nodes, ReLU, sum, and normal init.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu1_reinit")
def config_relu1_reinit() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    x = xor_data_centered()
    y = xor_labels_T1()
    model = models.Model_ReLU1()
    model.reinit_dead_data(model.init_normal, x, 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=800, 
                                stop_training_loss_threshold=1e-7,
                                loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=x, y=y),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold, convergence_analysis=False, save_plots=True, dead_data_analysis=True, mirror_pair_detection=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False, random_seeds=[18]),
        description="Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu1_reinit_margin")
def config_relu1_reinit_margin() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    x = xor_data_centered()
    y = xor_labels_T1()
    model = models.Model_ReLU1()
    model.reinit_dead_data(model.init_normal, x, 100, min_threshold=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=800, 
                                stop_training_loss_threshold=1e-7,
                                loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=x, y=y),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold, convergence_analysis=False, save_plots=True, dead_data_analysis=True, mirror_pair_detection=True),
        execution=ExecutionConfig(num_runs=500, skip_existing=False, random_seeds=[18]),
        description="Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu1_bhs")
def config_relu1_bhs() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    x = xor_data_centered()
    y = xor_labels_T1()
    model = models.Model_ReLU1()
    # model.init_bounded_hypersphere(model.init_normal, radius=1.4)

    model.reinit_dead_data(
        lambda : model.init_bounded_hypersphere(model.init_normal, radius=1.4), 
        x, 100, min_threshold=0.3)

    # print(f"x = {x}")
    # print(f"model.linear1.W = {model.linear1.weight }")
    # print(f"model.linear1.b = {model.linear1.bias}")
    # print(f"relu1_bhs init activation = {model.linear1(x)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.7, 0.9))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.8))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=2000, 
                                stop_training_loss_threshold=1e-7,
                                loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=x, y=y),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold, convergence_analysis=False, save_plots=True, dead_data_analysis=True, mirror_pair_detection=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False, random_seeds=[18]),
        description="Centered XOR with two nodes, ReLU, sum, and bounded hypersphere initialization with norm weights.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu1_monitor")
def config_relu1_monitor() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment with early-failure monitor."""

    # ------------------------------------------------------------------
    # 1) Build (and immediately inspect) the tiny XOR dataset so we know
    #    how many samples the monitor has to track.
    # ------------------------------------------------------------------
    x_data = xor_data_centered()          # shape: (4, 2)
    y_data = xor_labels_T1()              # shape: (4,)
    dataset_size = x_data.shape[0]        # == 4

    # ------------------------------------------------------------------
    # 2) Model, optimiser, loss are identical to before.
    # ------------------------------------------------------------------
    model = models.Model_ReLU1().init_normal()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    # ------------------------------------------------------------------
    # 3) Instantiate health monitors with dataset_size (+ optional
    #    patience and threshold — defaults are fine but shown for clarity).
    # ------------------------------------------------------------------
    # health_monitor = monitor.DeadSampleMonitor(
    #     model,
    #     dataset_size=dataset_size,  # required
    #     patience=3,                 # flag after 3 consecutive dead epochs
    #     classifier_threshold=0.5    # output ≥0.5 → class 1
    # )

    hook_manager = monitor.SharedHookManager(model)
    health_monitor = monitor.CompositeMonitor([
        monitor.DeadSampleMonitor(hook_manager, dataset_size=dataset_size, patience=5, classifier_threshold=0.5),
        monitor.BoundsMonitor(hook_manager, dataset_size=dataset_size, radius=1.5),
    ])

    # health_monitor = monitor.DeadSampleMonitor(hook_manager, dataset_size=dataset_size, patience=6, classifier_threshold=0.5)

    # ------------------------------------------------------------------
    # 4) Assemble the usual experiment config.
    # ------------------------------------------------------------------
    return ExperimentConfig(
        model=model,
        training=TrainingConfig(
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=2000,
            stop_training_loss_threshold=1e-7,
            loss_change_threshold=None,
            loss_change_patience=10,
            health_monitor=health_monitor,
        ),
        data=DataConfig(
            x=x_data,
            y=y_data
        ),
        analysis=AnalysisConfig(
            accuracy_fn=accuracy_binary_threshold, 
            convergence_analysis=False,
            save_plots=True,
            dead_data_analysis=True,
            mirror_pair_detection=True
        ),
        execution=ExecutionConfig(
            num_runs=1000,
            skip_existing=False,
            random_seeds=[18]
        ),
        description=(
            "Centered XOR with two nodes, ReLU, sum, normal init, "
            "and early-failure degeneracy detection."
        ),
        logging=LoggingConfig(train_epochs=50),
    )

@experiment("relu1_mirror")
def config_relu1_mirror() -> ExperimentConfig:
    """Factory function for ReLU XOR experiment."""
    model = models.Model_ReLU1().init_normal().init_mirror()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=800, 
                                stop_training_loss_threshold=1e-7,
                                loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_T1()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_binary_threshold, convergence_analysis=False, save_plots=False, dead_data_analysis=True, mirror_pair_detection=True, failure_angles=True),
        execution=ExecutionConfig(num_runs=1000, skip_existing=False, random_seeds=[18]),
        description="Centered XOR with two nodes, ReLU, sum, and mirrored normal init.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_bce")
def config_abs2_single_bce() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_bce_l2reg")
def abs2_single_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_mse")
def config_abs2_single_mse() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output MSE loss using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_mse_l2reg")
def config_abs2_single_mse_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.MSELoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output MSE loss, L2 reg, using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_bce_confidence")
def config_abs2_single_bce_confidence() -> ExperimentConfig:
    model = models.Model_Xor2_Confidence(middle=1, activation=models.Abs()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss using a single Abs unit. Includes a confidence final layer.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("abs2_single_bce_confidence_l2reg")
def abs2_abs2_single_bce_confidence_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2_Confidence(middle=1, activation=models.Abs()).init()

    optimizer_grouped_parameters = [
        {
            'params': itertools.chain(model.linear1.parameters(), model.linear2.parameters()),
            'weight_decay': 1e-1  # Apply L2 regularization here
        },
        {
            'params': model.confidence.parameters(),
            'weight_decay': 0.0             # Do NOT apply L2 regularization here
        }
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=True),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, using a single Abs unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu2_two_bce")
def config_relu2_two_bce() -> ExperimentConfig:
    model = models.Model_Xor2(middle=2, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss using a two ReLU units.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu2_two_bce_l2reg")
def config_relu2_two_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=2, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, and two ReLU units.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu2_one_bce_l2reg")
def config_relu2_one_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=1, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, and one ReLU unit.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu2_three_bce_l2reg")
def config_relu2_three_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=3, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, and three ReLU units.",
        logging=LoggingConfig(train_epochs=50)
    )

@experiment("relu2_four_bce_l2reg")
def config_relu2_four_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=4, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, and four ReLU units.",
        logging=LoggingConfig(train_epochs=50)
    )


@experiment("relu2_eight_bce_l2reg")
def config_relu2_eight_bce_l2reg() -> ExperimentConfig:
    model = models.Model_Xor2(middle=8, activation=nn.ReLU()).init()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-1)
    loss_function = nn.BCEWithLogitsLoss()

    return ExperimentConfig(
        model=model,
        training=TrainingConfig(optimizer=optimizer, loss_function=loss_function, epochs=5000, stop_training_loss_threshold=1e-7, loss_change_threshold=1e-24, loss_change_patience=10),
        data=DataConfig(x=xor_data_centered(), y=xor_labels_one_hot()),
        analysis=AnalysisConfig(accuracy_fn=accuracy_one_hot, save_plots=False),
        execution=ExecutionConfig(num_runs=50, skip_existing=False),
        description="Centered XOR with 2-output BCE loss, L2 reg, and eight ReLU units.",
        logging=LoggingConfig(train_epochs=50)
    )

