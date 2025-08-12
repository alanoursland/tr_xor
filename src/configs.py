# configs.py - Experiment Configuration for Prototype Surface Experiments

"""
Comprehensive experiment configuration system for prototype surface research.
Provides structured experiment definitions, validation, inheritance, and parameter sweep
capabilities. Designed to enable systematic investigation of prototype surface theory across different
model architectures, activation functions, and training configurations.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import torch

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


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""

    optimizer: torch.optim.Optimizer = None
    eps: float = 1e-8

    loss_function: torch.nn.Module = None
    regularizer_function: Optional[Callable] = None

    epochs: int = None
    batch_size: int = None

    # Training Monitor
    training_monitor: Optional[Any] = field(default=None)

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
   parameter_displacement: bool = False     # Initial→final weight angle/norm analysis
   distance_to_hyperplanes: bool = True     # Clusters linear layer distances to classes
   hyperplane_clustering: bool = False      # Clusters hyperplane positions
   mirror_weight_detection: bool = False    # Detect w_i ≈ -w_j pairs (ReLU-specific)
   failure_angle_analysis: bool = False     # Initial angle analysis for failed runs
   dead_data_analysis: bool = False         # ReLU-specific dead data detection
   dead_unit_analysis: bool = False         # ReLU-specific dead unit detection
   
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

