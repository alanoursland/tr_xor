# run.py - Experiment Execution for PSL Experiments

"""
Central orchestration system for running Prototype Surface Learning (PSL) experiments.
Provides command-line interface, experiment management, training coordination, and
comprehensive state management with logging and reproducibility features.
"""

import argparse
import logging
import time
import signal
import sys
import traceback
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib import setup_experiment_environment

# Import project modules
from configs import ExperimentConfig, get_experiment_config, list_experiments, apply_overrides
from models import create_xor_model, create_parity_model, create_custom_model
from data import generate_xor_data, generate_parity_data, create_training_batches
from utils import (
    set_global_random_seeds,
    generate_experiment_seeds,
    setup_experiment_logging,
    create_experiment_directory_structure,
    save_model_with_metadata,
    create_loss_function,
    setup_optimizer,
    create_learning_rate_scheduler,
    implement_early_stopping,
    detect_training_convergence,
)

# ==============================================================================
# Global State Management
# ==============================================================================


class ExperimentState:
    """Global state manager for experiment execution."""

    def __init__(self):
        self.current_experiment: Optional[str] = None
        self.current_run: Optional[int] = None
        self.interrupted: bool = False
        self.start_time: Optional[float] = None
        self.logger: Optional[logging.Logger] = None
        self.output_dirs: Optional[Dict[str, Path]] = None

    def reset(self) -> None:
        """Reset state for new experiment."""
        pass

    def set_experiment(self, name: str, run_id: int) -> None:
        """Set current experiment and run identifiers."""
        pass

    def signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals gracefully."""
        pass


# Global state instance
_experiment_state = ExperimentState()


# ==============================================================================
# Command Line Interface
# ==============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for experiment execution.

    Returns:
        Configured argument parser
    """
    pass


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        Parsed command line arguments
    """
    pass


def validate_cli_arguments(args: argparse.Namespace) -> Tuple[bool, List[str]]:
    """
    Validate command line arguments for consistency and completeness.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass


def setup_cli_logging(args: argparse.Namespace) -> None:
    """
    Setup initial logging for CLI operations.

    Args:
        args: Command line arguments containing logging preferences
    """
    pass


# ==============================================================================
# Experiment Initialization and Setup
# ==============================================================================


def load_and_validate_config(
    experiment_name: str, config_overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """
    Load experiment configuration and validate all parameters.

    Args:
        experiment_name: Name of experiment configuration to load
        config_overrides: Optional dictionary of configuration overrides

    Returns:
        Validated and complete experiment configuration
    """
    pass


def setup_output_directories(base_dir: Path, experiment_name: str, timestamp: bool = True) -> Dict[str, Path]:
    """
    Create structured output directory hierarchy for experiment results.

    Args:
        base_dir: Base directory for all experiment outputs
        experiment_name: Name of current experiment
        timestamp: Whether to include timestamp in directory name

    Returns:
        Dictionary mapping directory types to paths
    """
    pass


def initialize_logging(output_dir: Path, experiment_name: str, verbosity: str = "INFO") -> logging.Logger:
    """
    Initialize comprehensive logging system for experiment execution.

    Args:
        output_dir: Directory for log files
        experiment_name: Name of experiment for log file naming
        verbosity: Logging verbosity level

    Returns:
        Configured logger instance
    """
    pass


def setup_device_and_seeds(config: ExperimentConfig, run_id: int) -> Tuple[torch.device, int]:
    """
    Configure compute device and random seeds for reproducible execution.

    Args:
        config: Experiment configuration
        run_id: Current run identifier

    Returns:
        Tuple of (device, random_seed)
    """
    pass


def create_experiment_manifest(config: ExperimentConfig, output_dir: Path) -> None:
    """
    Create experiment manifest with complete configuration and metadata.

    Args:
        config: Complete experiment configuration
        output_dir: Output directory for manifest file
    """
    pass


# ==============================================================================
# Model and Data Instantiation
# ==============================================================================


def create_model_from_config(model_config: Any, device: torch.device) -> nn.Module:
    """
    Instantiate model from configuration specification.

    Args:
        model_config: Model configuration from experiment config
        device: Target device for model

    Returns:
        Instantiated and initialized model
    """
    pass


def create_dataset_from_config(data_config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset according to data configuration.

    Args:
        data_config: Data configuration from experiment config

    Returns:
        Tuple of (input_data, labels)
    """
    pass


def setup_training_components(training_config: Any, model: nn.Module) -> Dict[str, Any]:
    """
    Setup optimizer, loss function, scheduler, and other training components.

    Args:
        training_config: Training configuration from experiment config
        model: Model instance for training

    Returns:
        Dictionary containing all training components
    """
    pass


def initialize_model_weights(model: nn.Module, config: Any) -> None:
    """
    Initialize model weights according to configuration specification.

    Args:
        model: Model to initialize
        config: Model configuration with initialization parameters
    """
    pass


# ==============================================================================
# Training Loop Management
# ==============================================================================


class TrainingTracker:
    """Track training progress and manage training state."""

    def __init__(self, config: Any):
        self.config = config
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = []
        self.should_stop = False
        self.start_time = None

    def update(self, loss: float, accuracy: float, learning_rate: float) -> None:
        """Update training metrics and check stopping criteria."""
        pass

    def should_early_stop(self) -> bool:
        """Check if training should stop early."""
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        pass


def execute_training_run(
    model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
    training_components: Dict[str, Any],
    config: ExperimentConfig,
    run_id: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Execute single training run with comprehensive logging and monitoring.

    Args:
        model: Model to train
        data: Training data (inputs, labels)
        training_components: Optimizer, loss function, scheduler, etc.
        config: Complete experiment configuration
        run_id: Current run identifier
        device: Training device

    Returns:
        Tuple of (trained_model, training_statistics)
    """
    # Move model to device
    model = model.to(device)
    x, y = data
    x = x.to(device)
    y = y.to(device)
    # print(f"x.size = {x.size()}")
    # print(f"y.size = {y.size()}")

    # Extract training components
    optimizer = training_components["optimizer"]
    loss_function = training_components["loss_function"]

    # Training tracking
    losses = []
    start_time = time.time()
    best_loss = float("inf")

    # Training loop
    for epoch in range(config.training.epochs):
        model.train()

        # Forward pass
        outputs = model(x)
        loss = loss_function(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        current_loss = loss.item()
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss

        # Log progress occasionally
        if epoch % config.logging.train_epochs == 0 or epoch == config.training.epochs - 1:
            # Compute current accuracy
            with torch.no_grad():
                model.eval()
                preds = (outputs.squeeze() > 0.5).float()
                accuracy = (preds == y).float().mean().item()
                model.train()  # Switch back to training mode
            
            print(f"  Run {run_id}, Epoch {epoch:4d} | Loss: {current_loss:.6f} | Accuracy: {accuracy:.3f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(x)
        final_loss = loss_function(final_outputs, y).item()

        # For classification, compute accuracy
        if len(final_outputs.shape) > 1 and final_outputs.shape[1] > 1:
            # Multi-class output
            preds = torch.argmax(final_outputs, dim=1)
            accuracy = (preds == y).float().mean().item()
        else:
            # Single output (regression or binary)
            preds = (final_outputs.squeeze() > 0.5).float()
            accuracy = (preds == y.float()).float().mean().item()

    training_time = time.time() - start_time

    # Compile statistics
    stats = {
        "run_id": run_id,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "accuracy": accuracy,
        "training_time": training_time,
        "epochs_completed": config.training.epochs,
        "loss_history": losses,
    }

    return model, stats


def training_epoch(
    model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
    training_components: Dict[str, Any],
    tracker: TrainingTracker,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute single training epoch.

    Args:
        model: Model being trained
        data: Training data
        training_components: Training components (optimizer, loss, etc.)
        tracker: Training progress tracker
        device: Training device

    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    pass


def evaluate_model(
    model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor], loss_function: nn.Module, device: torch.device
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate model performance on given data.

    Args:
        model: Model to evaluate
        data: Evaluation data (inputs, labels)
        loss_function: Loss function for evaluation
        device: Evaluation device

    Returns:
        Tuple of (loss, accuracy, detailed_metrics)
    """
    pass


def save_training_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, tracker: TrainingTracker, checkpoint_dir: Path
) -> None:
    """
    Save training checkpoint for resumption capability.

    Args:
        model: Current model state
        optimizer: Current optimizer state
        tracker: Training progress tracker
        checkpoint_dir: Directory for checkpoint files
    """
    pass


def load_training_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: optim.Optimizer) -> TrainingTracker:
    """
    Load training checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to restore state to
        optimizer: Optimizer to restore state to

    Returns:
        Restored training tracker
    """
    pass


# ==============================================================================
# Multi-Run Coordination
# ==============================================================================


def coordinate_multiple_runs(config: ExperimentConfig, output_dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    Coordinate execution of multiple independent training runs.

    Args:
        config: Complete experiment configuration
        output_dirs: Dictionary of output directories

    Returns:
        List of results from all runs
    """
    pass


def execute_single_run(run_config: Tuple[ExperimentConfig, int, Dict[str, Path]]) -> Dict[str, Any]:
    """
    Execute single training run (for parallel execution).

    Args:
        run_config: Tuple of (config, run_id, output_dirs)

    Returns:
        Results dictionary for this run
    """
    pass


def execute_parallel_runs(config: ExperimentConfig, output_dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    Execute multiple runs in parallel using multiprocessing.

    Args:
        config: Complete experiment configuration
        output_dirs: Dictionary of output directories

    Returns:
        List of results from all parallel runs
    """
    pass


def aggregate_run_results(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics and metrics across multiple runs.

    Args:
        run_results: List of individual run results

    Returns:
        Aggregated statistics across all runs
    """
    pass


def detect_convergence_failures(run_results: List[Dict[str, Any]], failure_criteria: Dict[str, Any]) -> List[int]:
    """
    Identify runs that failed to converge or had other issues.

    Args:
        run_results: List of individual run results
        failure_criteria: Criteria for detecting failures

    Returns:
        List of run IDs that failed
    """
    pass


# ==============================================================================
# Progress Monitoring and Reporting
# ==============================================================================


def real_time_progress_display(
    current_run: int, total_runs: int, epoch: int, total_epochs: int, loss: float, accuracy: float
) -> None:
    """
    Display real-time training progress with dynamic updates.

    Args:
        current_run: Current run number
        total_runs: Total number of runs
        epoch: Current epoch
        total_epochs: Total epochs per run
        loss: Current loss value
        accuracy: Current accuracy
    """
    pass


def estimate_completion_time(start_time: float, current_progress: float, total_work: float) -> Tuple[float, str]:
    """
    Estimate remaining time based on current progress.

    Args:
        start_time: Experiment start timestamp
        current_progress: Current progress (0.0 to 1.0)
        total_work: Total amount of work (arbitrary units)

    Returns:
        Tuple of (estimated_seconds_remaining, formatted_time_string)
    """
    pass


def log_training_milestone(
    epoch: int, loss: float, accuracy: float, learning_rate: float, logger: logging.Logger
) -> None:
    """
    Log training milestone with structured formatting.

    Args:
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        learning_rate: Current learning rate
        logger: Logger instance
    """
    pass


def generate_training_summary(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Generate comprehensive training summary across all runs.

    Args:
        run_results: Results from all training runs
        config: Experiment configuration

    Returns:
        Comprehensive training summary
    """
    pass


def create_progress_report(current_state: Dict[str, Any], config: ExperimentConfig) -> str:
    """
    Create formatted progress report for logging and display.

    Args:
        current_state: Current experiment state
        config: Experiment configuration

    Returns:
        Formatted progress report string
    """
    pass


# ==============================================================================
# State Management and Persistence
# ==============================================================================


def save_run_state(
    run_id: int, model: nn.Module, optimizer: optim.Optimizer, epoch: int, losses: List[float], output_dir: Path
) -> None:
    """
    Save complete state for individual training run.

    Args:
        run_id: Run identifier
        model: Current model state
        optimizer: Current optimizer state
        epoch: Current epoch
        losses: Training loss history
        output_dir: Output directory for state files
    """
    pass


def load_run_state(state_path: Path) -> Dict[str, Any]:
    """
    Load saved training run state for resumption.

    Args:
        state_path: Path to saved state file

    Returns:
        Dictionary containing restored state
    """
    pass


def save_final_model(model: nn.Module, config: ExperimentConfig, stats: Dict[str, Any], output_path: Path) -> None:
    """
    Save final trained model with complete metadata and statistics.

    Args:
        model: Trained model to save
        config: Experiment configuration
        stats: Training statistics and metrics
        output_path: Path for saved model
    """
    pass


def create_experiment_manifest(config: ExperimentConfig, run_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Create comprehensive experiment manifest documenting all aspects of execution.

    Args:
        config: Complete experiment configuration
        run_results: Results from all training runs
        output_dir: Output directory for manifest
    """
    pass


def save_experiment_summary(aggregated_results: Dict[str, Any], config: ExperimentConfig, output_dir: Path) -> None:
    """
    Save aggregated experiment summary with statistics and analysis.

    Args:
        aggregated_results: Aggregated results across all runs
        config: Experiment configuration
        output_dir: Output directory for summary
    """
    pass


# ==============================================================================
# Error Handling and Recovery
# ==============================================================================


class ExperimentError(Exception):
    """Custom exception for experiment-related errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        run_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.run_id = run_id
        self.context = context or {}


def handle_training_error(error: Exception, run_id: int, config: ExperimentConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    Handle training errors with appropriate recovery strategies.

    Args:
        error: Exception that occurred during training
        run_id: Run identifier where error occurred
        config: Experiment configuration

    Returns:
        Tuple of (should_continue, error_info)
    """
    pass


def handle_configuration_error(error: Exception, config_name: str) -> None:
    """
    Handle configuration-related errors with helpful messages.

    Args:
        error: Configuration error that occurred
        config_name: Name of problematic configuration
    """
    pass


def handle_resource_error(error: Exception, config: ExperimentConfig) -> Tuple[bool, ExperimentConfig]:
    """
    Handle resource-related errors (memory, GPU, etc.) with fallback strategies.

    Args:
        error: Resource error that occurred
        config: Current experiment configuration

    Returns:
        Tuple of (can_continue, modified_config)
    """
    pass


def cleanup_on_interruption(signal_num: int) -> None:
    """
    Perform cleanup operations when experiment is interrupted.

    Args:
        signal_num: Signal number that caused interruption
    """
    pass


def attempt_error_recovery(error: Exception, context: Dict[str, Any]) -> bool:
    """
    Attempt to recover from errors using various strategies.

    Args:
        error: Error to attempt recovery from
        context: Context information for recovery

    Returns:
        True if recovery was successful
    """
    pass


# ==============================================================================
# Main Execution Functions
# ==============================================================================


def run_experiment(
    experiment_name: str,
    num_runs: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute complete experiment with specified configuration.

    Args:
        experiment_name: Name of experiment configuration to run
        num_runs: Override number of runs (None uses config default)
        device: Override device specification (None uses config default)
        verbose: Enable verbose logging output
        config_overrides: Optional configuration overrides

    Returns:
        Complete experiment results and summary
    """
    # Load configuration
    config = get_experiment_config(experiment_name)

    # Apply overrides
    if num_runs is not None:
        config.execution.num_runs = num_runs
    if device is not None:
        config.execution.device = device
    if config_overrides:
        config = apply_overrides(config, config_overrides)

    # Setup environment
    setup_info = setup_experiment_environment(
        experiment_name=experiment_name, seed=42, device=config.execution.device  # Could get from config
    )

    logger = setup_info["logger"]
    output_dirs = setup_info["output_dirs"]
    actual_device = setup_info["device"]

    # Run multiple training runs
    all_results = []
    successful_runs = 0

    for run_id in range(config.execution.num_runs):
        if verbose:
            print(f"Starting run {run_id + 1}/{config.execution.num_runs}")

        try:
            # Create fresh model for each run (call config factory again)
            fresh_config = get_experiment_config(experiment_name)

            # Debug: Check what we actually got
            if fresh_config.model is None:
                raise ValueError(f"Config factory returned None model for experiment '{experiment_name}'")

            # Execute single run
            model, run_result = execute_training_run(
                model=fresh_config.model,
                data=(config.data.x, config.data.y),
                training_components={
                    "optimizer": fresh_config.training.optimizer,
                    "loss_function": fresh_config.training.loss_function,
                },
                config=fresh_config,
                run_id=run_id,
                device=actual_device,
            )

            all_results.append(run_result)
            successful_runs += 1

            # Save each run as it completes
            run_dir = output_dirs["experiment"] / "runs" / f"{run_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), run_dir / "model.pt")
            
            # Save run statistics and config
            torch.save(run_result, run_dir / "stats.pt")

            # Save minimal config info (no data)
            config_summary = {
                "experiment_name": experiment_name,
                "epochs": fresh_config.training.epochs,
                "model_type": type(fresh_config.model).__name__,
                "optimizer_type": type(fresh_config.training.optimizer).__name__,
                "loss_function_type": type(fresh_config.training.loss_function).__name__,
                "description": fresh_config.description
            }
            torch.save(config_summary, run_dir / "config_summary.pt")
            
            if verbose:
                print(f"✓ Run {run_id + 1} saved to {run_dir}")
            
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Run {run_id} failed with {type(e).__name__}: {e}")
            logger.error(f"Full traceback:\n{error_details}")
            if verbose:
                print(f"✗ Run {run_id + 1} failed: {type(e).__name__}: {e}")
                print(f"Traceback:\n{error_details}")
        finally:
            # Always clean up, even if there was an error
            if 'fresh_config' in locals():
                fresh_config.cleanup()
                del fresh_config

    # Aggregate results
    summary = {
        "total_runs": config.execution.num_runs,
        "successful_runs": successful_runs,
        "avg_final_loss": sum(r["final_loss"] for r in all_results) / len(all_results) if all_results else 0,
        "avg_accuracy": sum(r.get("accuracy", 0) for r in all_results) / len(all_results) if all_results else 0,
        "total_time": sum(r["training_time"] for r in all_results),
    }


    # Print detailed run summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RUN SUMMARY")
    print("=" * 60)
    
    # Accuracy distribution - XOR specific (only 0%, 25%, 50%, 75%, 100%)
    accuracies = [r.get("accuracy", 0) for r in all_results]
    acc_counts = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
    
    for acc in accuracies:
        if acc <= 0.125:  # Round 0% to 0.0
            acc_counts[0.0] += 1
        elif acc <= 0.375:  # Round 25% to 0.25
            acc_counts[0.25] += 1
        elif acc <= 0.625:  # Round 50% to 0.5
            acc_counts[0.5] += 1
        elif acc <= 0.875:  # Round 75% to 0.75
            acc_counts[0.75] += 1
        else:  # Round 100% to 1.0
            acc_counts[1.0] += 1

    print(f"Accuracy Distribution:")
    print(f"  100% (4/4 correct): {acc_counts[1.0]:2d} runs ({acc_counts[1.0]/len(all_results)*100:.1f}%)")
    print(f"   75% (3/4 correct): {acc_counts[0.75]:2d} runs ({acc_counts[0.75]/len(all_results)*100:.1f}%)")
    print(f"   50% (2/4 correct): {acc_counts[0.5]:2d} runs ({acc_counts[0.5]/len(all_results)*100:.1f}%)")
    print(f"   25% (1/4 correct): {acc_counts[0.25]:2d} runs ({acc_counts[0.25]/len(all_results)*100:.1f}%)")
    print(f"    0% (0/4 correct): {acc_counts[0.0]:2d} runs ({acc_counts[0.0]/len(all_results)*100:.1f}%)")
    
    # Loss distribution  
    final_losses = [r["final_loss"] for r in all_results]
    converged_runs = sum(1 for loss in final_losses if loss < 0.01)
    
    print(f"\nConvergence:")
    print(f"  Converged (<0.01):  {converged_runs:2d} runs ({converged_runs/len(all_results)*100:.1f}%)")
    print(f"  Best final loss:    {min(final_losses):.6f}")
    print(f"  Worst final loss:   {max(final_losses):.6f}")
    
    print(f"\nTiming:")
    print(f"  Total time:         {summary['total_time']:.2f}s")
    print(f"  Average per run:    {summary['total_time']/len(all_results):.2f}s")
    
    print("=" * 60)

    # Save overall experiment statistics to JSON
    stats_file = output_dirs["experiment"] / "stats.json"

    experiment_stats = {
        "experiment_name": experiment_name,
        "total_runs": summary["total_runs"],
        "successful_runs": summary["successful_runs"],
        "avg_final_loss": summary["avg_final_loss"],
        "avg_accuracy": summary["avg_accuracy"],
        "total_time": summary["total_time"],
        "avg_time_per_run": summary["total_time"] / summary["total_runs"],
        
        # Accuracy distribution
        "accuracy_distribution": {
            "100_percent": acc_counts[1.0],
            "75_percent": acc_counts[0.75], 
            "50_percent": acc_counts[0.5],
            "25_percent": acc_counts[0.25],
            "0_percent": acc_counts[0.0]
        },
        
        # Convergence stats
        "convergence": {
            "converged_runs": converged_runs,
            "convergence_rate": converged_runs / len(all_results),
            "best_final_loss": min(final_losses),
            "worst_final_loss": max(final_losses)
        },
        
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_description": config.description
    }

    with open(stats_file, 'w') as f:
        json.dump(experiment_stats, f, indent=2)

    if verbose:
        print(f"Experiment statistics saved to: {stats_file}")

    return {
        "experiment_name": experiment_name,
        "summary": summary,
        "individual_runs": all_results,
        "output_dir": setup_info["output_dirs"]["experiment"],
        "config": config,
    }


def run_experiment_batch(
    experiment_list: List[str], parallel: bool = False, shared_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Execute batch of experiments sequentially or in parallel.

    Args:
        experiment_list: List of experiment names to execute
        parallel: Whether to run experiments in parallel
        shared_config: Configuration overrides applied to all experiments

    Returns:
        Dictionary mapping experiment names to results
    """
    pass


def run_parameter_sweep(
    base_experiment: str, param_grid: Dict[str, List[Any]], sweep_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute parameter sweep across grid of hyperparameter values.

    Args:
        base_experiment: Base experiment configuration for sweep
        param_grid: Dictionary mapping parameter names to value lists
        sweep_name: Optional name for sweep (for organization)

    Returns:
        Results from all parameter combinations
    """
    pass


def resume_experiment(experiment_dir: Path, resume_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resume interrupted experiment from saved state.

    Args:
        experiment_dir: Directory containing interrupted experiment
        resume_config: Optional configuration modifications for resumption

    Returns:
        Results from resumed experiment
    """
    pass


def dry_run_experiment(experiment_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform dry run to validate configuration without actual training.

    Args:
        experiment_name: Name of experiment configuration to validate
        config_overrides: Optional configuration overrides

    Returns:
        Validation results and estimated resource requirements
    """
    pass


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> int:
    """
    Main entry point for experiment execution script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command line arguments
        if len(sys.argv) != 2:
            print("Usage: python run.py <experiment_name>")
            print("\nAvailable experiments:")
            available_experiments = list_experiments()
            for exp in available_experiments:
                print(f"  - {exp}")
            return 1

        experiment_name = sys.argv[1]

        print(f"Starting experiment: {experiment_name}")
        print("=" * 50)

        # Load and validate experiment configuration
        print("Loading experiment configuration...")
        try:
            config = get_experiment_config(experiment_name)
            print(f"✓ Configuration loaded successfully")
        except KeyError:
            print(f"✗ Unknown experiment: {experiment_name}")
            print("\nAvailable experiments:")
            available_experiments = list_experiments()
            for exp in available_experiments:
                print(f"  - {exp}")
            return 1
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            return 1

        # Setup experiment environment
        print("Setting up experiment environment...")
        setup_info = setup_experiment_environment(
            experiment_name=experiment_name,
            seed=config.execution.random_seeds[0] if config.execution.random_seeds else 42,
            device=config.execution.device,
        )
        print(f"✓ Environment setup complete (device: {setup_info['device']})")

        # Run the experiment
        print(f"Running experiment with {config.execution.num_runs} runs...")
        print("-" * 30)

        results = run_experiment(
            experiment_name=experiment_name,
            num_runs=config.execution.num_runs,
            device=config.execution.device,
            verbose=True,
        )

        print("-" * 30)
        print("✓ Experiment completed successfully")

        # Print summary results
        if "summary" in results:
            summary = results["summary"]
            print(f"\nResults Summary:")
            print(f"  Total runs: {summary.get('total_runs', 'N/A')}")
            print(f"  Successful runs: {summary.get('successful_runs', 'N/A')}")
            print(f"  Average final loss: {summary.get('avg_final_loss', 'N/A'):.6f}")
            print(f"  Average accuracy: {summary.get('avg_accuracy', 'N/A'):.4f}")
            print(f"  Total time: {summary.get('total_time', 'N/A'):.2f}s")

        print(f"\nResults saved to: {results.get('output_dir', 'N/A')}")
        print("=" * 50)
        print("Experiment completed successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n✗ Experiment interrupted by user")
        cleanup_on_interruption(signal.SIGINT)
        return 130

    except Exception as e:
        print(f"\n✗ Unexpected error during experiment execution:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return 1


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful interruption handling."""

    def signal_handler(signum, frame):
        print(f"\n⚠️  Received signal {signum}")
        if signum == signal.SIGINT:
            print("Interrupt signal received (Ctrl+C)")
        elif signum == signal.SIGTERM:
            print("Termination signal received")

        print("Attempting graceful shutdown...")
        _experiment_state.interrupted = True
        _experiment_state.signal_handler(signum, frame)

        # Give a moment for cleanup, then exit
        time.sleep(0.5)
        sys.exit(130 if signum == signal.SIGINT else 1)

    # Register handlers for common signals
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

    # On Windows, also handle Ctrl+Break
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


def print_experiment_banner(experiment_name: str, config: ExperimentConfig) -> None:
    """
    Print formatted banner with experiment information.

    Args:
        experiment_name: Name of experiment being executed
        config: Experiment configuration
    """
    pass


def print_completion_summary(results: Dict[str, Any], execution_time: float) -> None:
    """
    Print formatted summary of experiment completion.

    Args:
        results: Experiment results
        execution_time: Total execution time in seconds
    """
    pass


# ==============================================================================
# Utility Functions for CLI
# ==============================================================================


def list_available_experiments() -> None:
    """Print formatted list of available experiment configurations."""
    pass


def show_experiment_info(experiment_name: str, detailed: bool = False) -> None:
    """
    Show information about specific experiment configuration.

    Args:
        experiment_name: Name of experiment to show info for
        detailed: Whether to show detailed configuration
    """
    pass


def estimate_resource_requirements(experiment_name: str) -> None:
    """
    Estimate computational resource requirements for experiment.

    Args:
        experiment_name: Name of experiment to estimate for
    """
    pass


if __name__ == "__main__":
    setup_signal_handlers()
    exit_code = main()
    sys.exit(exit_code)
