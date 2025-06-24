# run.py - Experiment Execution for Prototype Surface Experiments

"""
Central orchestration system for running prototype surface experiments.
Provides command-line interface, experiment management, training coordination, and
comprehensive state management with logging and reproducibility features.
"""

import logging
import time
import signal
import sys
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json
import torch
import torch.nn as nn
from lib import setup_experiment_environment
import numpy as np
import traceback

# Import project modules
from configs import ExperimentConfig, get_experiment_config, list_experiments

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
# Training Loop Management
# ==============================================================================

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
    # print(f"x = {x}")
    # print(f"y = {y}")
    # print(f"x.size = {x.size()}")
    # print(f"y.size = {y.size()}")
    # print(model)

    # Extract training components
    optimizer = training_components["optimizer"]
    loss_function = training_components["loss_function"]

    if config.training.health_monitor:
        config.training.health_monitor.to(device)

    # Training tracking
    losses = []
    start_time = time.time()
    best_loss = float("inf")
    patience_counter = 0
    loss_change_threshold = config.training.loss_change_threshold
    loss_change_patience = config.training.loss_change_patience

    # Training loop
    for epoch in range(config.training.epochs):
        model.train()

        # Forward pass
        outputs = model(x)
        loss = loss_function(outputs, y)

        # # monitor diagnostics
        # if config.training.health_monitor:    
        #     config.training.health_monitor.compute_per_example_gradients(x, y, config.training.loss_function)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        health_monitor = config.training.health_monitor
        if health_monitor:    
            batch_idx = list(range(y.size(0)))
            if not health_monitor.check(x, y, batch_idx):
                health_monitor.fix(x, y, batch_idx)
            # print(f"[epoch {epoch}] dead_data={metrics.dead_data_fraction:.2f}, torque={metrics.torque_ratio:.4f}, bias_drift={metrics.bias_drift:.2e}")
            # config.training.health_monitor.log_registered_state()
            # # Optional: respond to specific issues
            # if metrics.has_dead_data_issue:
            #     print(f"⚠️  Dead data detected: {metrics.dead_data_fraction:.1%}")
            # if metrics.has_torque_issue:
            #     print(f"⚠️  Low torque ratio: {metrics.torque_ratio:.4f}")
            # if metrics.has_bias_freeze_issue:
            #     print(f"⚠️  Bias drift too small: {metrics.bias_drift:.2e}")

        optimizer.step()

        # Track loss
        current_loss = loss.item()
        losses.append(current_loss)

        # Early exit if loss is low enough
        if config.training.stop_training_loss_threshold is not None:
            if current_loss < config.training.stop_training_loss_threshold:
                if config.logging.train_epochs > 0:
                    print(f"  Early stopping at epoch {epoch} (loss {current_loss:.6f} <= {config.training.stop_training_loss_threshold})")
                break

        # Early exit if model isn't improving.
        if loss_change_threshold is not None and loss_change_patience:
            loss_delta = best_loss - current_loss
            # print(f"loss_delta = {loss_delta}, patience_counter = {patience_counter}")
            if loss_delta > loss_change_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > loss_change_patience:
                print(f"  Convergence-based early stopping at epoch {epoch} "
                      f"(loss did not improve by {loss_change_threshold} for {loss_change_patience} steps)")
                break

        if current_loss < best_loss:
            best_loss = current_loss

        # Log progress occasionally
        if epoch % config.logging.train_epochs == 0 or epoch == config.training.epochs - 1:
            # Compute current accuracy
            with torch.no_grad():
                model.eval()
                if config.analysis.accuracy_fn is not None:
                    accuracy = config.analysis.accuracy_fn(outputs, y)
                else:
                    accuracy = 0.0
                model.train()  # Switch back to training mode
            
            print(f"  Run {run_id}, Epoch {epoch:4d} | Loss: {current_loss:.6f} | Accuracy: {accuracy:.3f}")
            # print(f"    W={model.linear1.weight}, b={model.linear1.bias}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(x)
        final_loss = loss_function(final_outputs, y).item()

        # For classification, compute accuracy
        if config.analysis.accuracy_fn is not None:
            accuracy = config.analysis.accuracy_fn(final_outputs, y)
        else:
            accuracy = 0.0

    training_time = time.time() - start_time

    # Compile statistics
    stats = {
        "run_id": run_id,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "accuracy": accuracy,
        "training_time": training_time,
        "epochs_completed": epoch + 1,
        "loss_history": losses,
    }

    return model, stats


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


# ==============================================================================
# Main Execution Functions
# ==============================================================================


def run_experiment(
    setup_info: Dict[str, Any],
    num_runs: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
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
    experiment_name = setup_info["experiment_name"]
    
    # Load configuration
    config = get_experiment_config(experiment_name)
    loss_change_threshold = config.training.loss_change_threshold

    # Apply overrides
    if num_runs is not None:
        config.execution.num_runs = num_runs
    if device is not None:
        config.execution.device = device

    logger = setup_info["logger"]
    output_dirs = setup_info["output_dirs"]
    actual_device = setup_info["device"]

    # Run multiple training runs
    all_results = []

    for run_id in range(config.execution.num_runs):
        if verbose:
            print(f"Starting run {run_id}/{config.execution.num_runs}")

        try:
            # Create fresh model for each run (call config factory again)
            fresh_config = get_experiment_config(experiment_name)

            # Debug: Check what we actually got
            if fresh_config.model is None:
                raise ValueError(f"Config factory returned None model for experiment '{experiment_name}'")

            # Create run directory before training
            run_dir = output_dirs["experiment"] / "runs" / f"{run_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save initial (untrained) model state
            torch.save(fresh_config.model.state_dict(), run_dir / "model_init.pt")

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
                print(f"✓ Run {run_id} saved to {run_dir}")
            
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Run {run_id} failed with {type(e).__name__}: {e}")
            logger.error(f"Full traceback:\n{error_details}")
            if verbose:
                print(f"✗ Run {run_id} failed: {type(e).__name__}: {e}")
                print(f"Traceback:\n{error_details}")
        finally:
            # Always clean up, even if there was an error
            if 'fresh_config' in locals():
                fresh_config.cleanup()
                del fresh_config

    # Aggregate results
    summary = {
        "total_runs": config.execution.num_runs,
        "avg_final_loss": sum(r["final_loss"] for r in all_results) / len(all_results) if all_results else 0,
        "avg_accuracy": sum(r.get("accuracy", 0) for r in all_results) / len(all_results) if all_results else 0,
        "total_time": sum(r["training_time"] for r in all_results),
    }

    # calculating epoch quantiles
    epoch_counts = [r.get("epochs_completed", 0) for r in all_results]
    if epoch_counts:
        avg_epochs = np.mean(epoch_counts)
        q25, q50, q75 = np.percentile(epoch_counts, [25, 50, 75])
        min_epochs = np.min(epoch_counts)
        max_epochs = np.max(epoch_counts)
    else:
        avg_epochs, q25, q50, q75 = 0, 0, 0, 0


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
    converged_runs = None
    converged_runs_percent = None
    if (config.training.stop_training_loss_threshold):
        converged_runs = sum(1 for loss in final_losses if loss < config.training.stop_training_loss_threshold)
        converged_runs_percent = converged_runs/len(all_results)*100
    
    print(f"\nEpoch Statistics:")
    print(f"  Mean epochs completed: {avg_epochs:.1f}")
    print(f"  Quantiles: {min_epochs:.0f} {q25:.0f} {q50:.0f} {q75:.0f} {max_epochs:.0f}")

    print(f"\nConvergence:")
    print(f"  Converged (<{config.training.stop_training_loss_threshold}):  {converged_runs:2d} runs ({converged_runs_percent:.1f}%)")
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
            traceback.print_exc()
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
            setup_info,
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
            print(f"  Average final loss: {summary.get('avg_final_loss', 'N/A'):.6f}")
            print(f"  Average accuracy: {summary.get('avg_accuracy', 'N/A'):.4f}")
            print(f"  Total time: {summary.get('total_time', 'N/A'):.2f}s")

        print(f"\nResults saved to: {results.get('output_dir', 'N/A')}")
        print("=" * 50)
        print("Experiment completed successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n✗ Experiment interrupted by user")
        return 130

    except Exception as e:
        print(f"\n✗ Unexpected error during experiment execution:")
        print(f"  {type(e).__name__}: {e}")

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



if __name__ == "__main__":
    setup_signal_handlers()
    exit_code = main()
    sys.exit(exit_code)
