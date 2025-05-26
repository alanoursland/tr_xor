# configs.py - Experiment Configuration for PSL Experiments

"""
Comprehensive experiment configuration system for Prototype Surface Learning (PSL) research.
Provides structured experiment definitions, validation, inheritance, and parameter sweep
capabilities. Designed to enable systematic investigation of PSL theory across different
model architectures, activation functions, and training configurations.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import copy


# ==============================================================================
# Configuration Schema and Types
# ==============================================================================

class ExperimentType(Enum):
    """Types of experiments supported by the framework."""
    XOR = "xor"
    PARITY = "parity"
    CUSTOM_BOOLEAN = "custom_boolean"
    SYNTHETIC = "synthetic"
    COMPARATIVE = "comparative"


class OptimizationType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"
    LBFGS = "lbfgs"


class LossType(Enum):
    """Supported loss function types."""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    BCE = "binary_cross_entropy"
    FOCAL = "focal"
    CUSTOM = "custom"


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    NONE = "none"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    CYCLIC = "cyclic"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    architecture: str  # "mlp", "xor_net", "minimal_xor", "symmetric_xor", "dual_path_xor"
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str  # From ActivationType enum
    initialization: str  # From InitializationType enum
    bias: bool = True
    dropout: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    
    # PSL-specific parameters
    enforce_symmetry: bool = False
    learnable_activation_params: bool = False
    prototype_aware_init: bool = False
    prototype_points: Optional[List[List[float]]] = None


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""
    optimizer: OptimizationType
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam
    eps: float = 1e-8
    
    loss_function: LossType
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    epochs: int
    batch_size: int
    
    # Learning rate scheduling
    scheduler: SchedulerType = SchedulerType.NONE
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 50
    min_delta: float = 1e-6
    restore_best_weights: bool = True
    
    # Convergence criteria
    convergence_threshold: float = 1e-6
    convergence_window: int = 10
    
    # Gradient clipping
    gradient_clipping: bool = False
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""
    problem_type: ExperimentType
    
    # XOR-specific
    xor_normalized: bool = True
    xor_center_origin: bool = True
    
    # Parity-specific
    parity_n_bits: int = 3
    parity_signed: bool = True
    parity_complete: bool = True  # Use complete truth table vs sampling
    parity_num_samples: Optional[int] = None  # For sampling
    
    # Data augmentation
    add_noise: bool = False
    noise_std: float = 0.1
    rotation_augment: bool = False
    rotation_angles: List[float] = field(default_factory=list)
    scaling_augment: bool = False
    scale_factors: List[float] = field(default_factory=list)
    
    # Preprocessing
    normalization: str = "none"  # From NormalizationType enum
    center_data: bool = False
    
    # Validation split
    validation_split: float = 0.0
    stratified_split: bool = True
    
    # Custom dataset parameters
    custom_truth_table: Optional[Dict[Tuple[int, ...], int]] = None


@dataclass
class AnalysisConfig:
    """Configuration for post-training analysis."""
    # Geometric analysis
    geometric_analysis: bool = True
    hyperplane_plots: bool = True
    prototype_region_analysis: bool = True
    decision_boundary_analysis: bool = True
    distance_field_analysis: bool = True
    
    # Weight analysis
    weight_analysis: bool = True
    mirror_pair_detection: bool = True
    weight_evolution_tracking: bool = False
    weight_clustering: bool = False
    symmetry_analysis: bool = True
    
    # Activation analysis
    activation_analysis: bool = True
    activation_patterns: bool = True
    sparsity_analysis: bool = True
    zero_activation_regions: bool = True
    
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
    
    # PSL-specific analysis
    prototype_surface_validation: bool = True
    separation_order_analysis: bool = True
    minsky_papert_metrics: bool = True


@dataclass
class ExecutionConfig:
    """Configuration for experiment execution."""
    num_runs: int = 10
    random_seeds: Optional[List[int]] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    
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
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    analysis: AnalysisConfig
    execution: ExecutionConfig
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_by: str = "psl_framework"
    notes: str = ""
    
    # Dependencies and inheritance
    base_config: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Pre-defined Experiment Templates
# ==============================================================================

# Base configurations for common scenarios
BASE_CONFIGS: Dict[str, ExperimentConfig] = {}

# XOR experiment templates
XOR_EXPERIMENTS: Dict[str, ExperimentConfig] = {}

# Parity experiment templates  
PARITY_EXPERIMENTS: Dict[str, ExperimentConfig] = {}

# Comparative analysis templates
COMPARATIVE_EXPERIMENTS: Dict[str, ExperimentConfig] = {}

# Parameter sweep templates
PARAMETER_SWEEPS: Dict[str, Dict[str, Any]] = {}


def _create_base_configs() -> None:
    """Create base configuration templates."""
    pass


def _create_xor_experiments() -> None:
    """Create XOR-specific experiment configurations."""
    pass


def _create_parity_experiments() -> None:
    """Create parity problem experiment configurations."""
    pass


def _create_comparative_experiments() -> None:
    """Create comparative analysis experiment configurations."""
    pass


def _create_parameter_sweeps() -> None:
    """Create parameter sweep configurations."""
    pass


# ==============================================================================
# Main Experiments Registry
# ==============================================================================

EXPERIMENTS: Dict[str, ExperimentConfig] = {}


def initialize_experiments() -> None:
    """Initialize all experiment configurations."""
    pass


def get_experiment_config(name: str) -> ExperimentConfig:
    """
    Retrieve experiment configuration by name.
    
    Args:
        name: Name of experiment configuration
        
    Returns:
        Complete experiment configuration
    """
    pass


def list_experiments(category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[str]:
    """
    List available experiment configurations.
    
    Args:
        category: Filter by experiment category
        tags: Filter by experiment tags
        
    Returns:
        List of experiment names matching criteria
    """
    pass


def get_experiment_categories() -> Dict[str, List[str]]:
    """
    Get experiment names organized by category.
    
    Returns:
        Dictionary mapping categories to experiment lists
    """
    pass


# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_experiment_config(config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """
    Validate experiment configuration for completeness and consistency.
    
    Args:
        config: Experiment configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def validate_model_config(config: ModelConfig) -> Tuple[bool, List[str]]:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def validate_training_config(config: TrainingConfig) -> Tuple[bool, List[str]]:
    """
    Validate training configuration parameters.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def validate_data_config(config: DataConfig) -> Tuple[bool, List[str]]:
    """
    Validate data configuration parameters.
    
    Args:
        config: Data configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def validate_analysis_config(config: AnalysisConfig) -> Tuple[bool, List[str]]:
    """
    Validate analysis configuration parameters.
    
    Args:
        config: Analysis configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def validate_execution_config(config: ExecutionConfig) -> Tuple[bool, List[str]]:
    """
    Validate execution configuration parameters.
    
    Args:
        config: Execution configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def check_config_compatibility(config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """
    Check for compatibility between different configuration sections.
    
    Args:
        config: Complete experiment configuration
        
    Returns:
        Tuple of (is_compatible, warning_messages)
    """
    pass


# ==============================================================================
# Configuration Inheritance and Merging
# ==============================================================================

def resolve_config_inheritance(config_name: str) -> ExperimentConfig:
    """
    Resolve configuration inheritance and apply overrides.
    
    Args:
        config_name: Name of configuration with potential inheritance
        
    Returns:
        Fully resolved configuration
    """
    pass


def merge_configs(base: ExperimentConfig, override: ExperimentConfig) -> ExperimentConfig:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    pass


def apply_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    """
    Apply dictionary overrides to configuration.
    
    Args:
        config: Base configuration
        overrides: Dictionary of override values
        
    Returns:
        Configuration with overrides applied
    """
    pass


def create_config_variant(base_config_name: str, modifications: Dict[str, Any], 
                         new_name: str) -> ExperimentConfig:
    """
    Create variant of existing configuration with modifications.
    
    Args:
        base_config_name: Name of base configuration
        modifications: Dictionary of modifications to apply
        new_name: Name for new variant configuration
        
    Returns:
        New variant configuration
    """
    pass


# ==============================================================================
# Parameter Sweep Generation
# ==============================================================================

def generate_parameter_sweep(base_config: ExperimentConfig, 
                           sweep_params: Dict[str, List[Any]]) -> List[ExperimentConfig]:
    """
    Generate grid search configurations from parameter sweep specification.
    
    Args:
        base_config: Base configuration for sweep
        sweep_params: Dictionary mapping parameter paths to value lists
        
    Returns:
        List of configurations for grid search
    """
    pass


def generate_random_search(base_config: ExperimentConfig,
                          param_distributions: Dict[str, Any],
                          num_samples: int) -> List[ExperimentConfig]:
    """
    Generate random search configurations from parameter distributions.
    
    Args:
        base_config: Base configuration for search
        param_distributions: Dictionary mapping parameters to distributions
        num_samples: Number of random configurations to generate
        
    Returns:
        List of configurations for random search
    """
    pass


def create_ablation_study(base_config: ExperimentConfig,
                         ablation_components: List[str]) -> List[ExperimentConfig]:
    """
    Create ablation study configurations by systematically removing components.
    
    Args:
        base_config: Complete configuration for ablation
        ablation_components: List of components to systematically remove
        
    Returns:
        List of ablation configurations
    """
    pass


# ==============================================================================
# Configuration I/O and Serialization
# ==============================================================================

def save_config(config: ExperimentConfig, filepath: Path, format: str = "yaml") -> None:
    """
    Save configuration to file in specified format.
    
    Args:
        config: Configuration to save
        filepath: Output file path
        format: File format ("yaml", "json", "toml")
    """
    pass


def load_config(filepath: Path, format: str = "auto") -> ExperimentConfig:
    """
    Load configuration from file with automatic format detection.
    
    Args:
        filepath: Path to configuration file
        format: File format ("yaml", "json", "toml", "auto")
        
    Returns:
        Loaded experiment configuration
    """
    pass


def export_config_template(config_type: str = "basic", filepath: Optional[Path] = None) -> str:
    """
    Export configuration template for manual editing.
    
    Args:
        config_type: Type of template ("basic", "advanced", "xor", "parity")
        filepath: Optional file path to save template
        
    Returns:
        Configuration template as string
    """
    pass


def import_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """
    Import configuration from dictionary format.
    
    Args:
        config_dict: Configuration as nested dictionary
        
    Returns:
        Structured experiment configuration
    """
    pass


def export_config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Export configuration to dictionary format.
    
    Args:
        config: Configuration to export
        
    Returns:
        Configuration as nested dictionary
    """
    pass


# ==============================================================================
# Configuration Analysis and Comparison
# ==============================================================================

def compare_configs(config1: ExperimentConfig, config2: ExperimentConfig) -> Dict[str, Any]:
    """
    Compare two configurations and identify differences.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary describing differences
    """
    pass


def analyze_config_space(configs: List[ExperimentConfig]) -> Dict[str, Any]:
    """
    Analyze configuration space covered by list of configurations.
    
    Args:
        configs: List of configurations to analyze
        
    Returns:
        Analysis of configuration space coverage
    """
    pass


def suggest_missing_experiments(existing_configs: List[ExperimentConfig],
                              coverage_criteria: Dict[str, Any]) -> List[ExperimentConfig]:
    """
    Suggest additional experiments to improve coverage of configuration space.
    
    Args:
        existing_configs: Currently available configurations
        coverage_criteria: Criteria for comprehensive coverage
        
    Returns:
        List of suggested additional configurations
    """
    pass


def generate_config_summary(config: ExperimentConfig) -> str:
    """
    Generate human-readable summary of configuration.
    
    Args:
        config: Configuration to summarize
        
    Returns:
        Summary string describing key configuration elements
    """
    pass


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_config_hash(config: ExperimentConfig) -> str:
    """
    Generate unique hash for configuration identification.
    
    Args:
        config: Configuration to hash
        
    Returns:
        Unique hash string
    """
    pass


def clone_config(config: ExperimentConfig) -> ExperimentConfig:
    """
    Create deep copy of configuration.
    
    Args:
        config: Configuration to clone
        
    Returns:
        Independent copy of configuration
    """
    pass


def update_config_metadata(config: ExperimentConfig, **metadata) -> ExperimentConfig:
    """
    Update configuration metadata fields.
    
    Args:
        config: Configuration to update
        **metadata: Metadata fields to update
        
    Returns:
        Configuration with updated metadata
    """
    pass


def get_config_dependencies(config: ExperimentConfig) -> List[str]:
    """
    Get list of dependencies for configuration (base configs, etc.).
    
    Args:
        config: Configuration to analyze
        
    Returns:
        List of dependency names
    """
    pass


# Initialize configurations on module import
initialize_experiments()