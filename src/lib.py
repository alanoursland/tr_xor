# lib.py - Common Library Functions for Prototype Surface Experiments

"""
Common library functions and utilities for prototype surface experiments.
Provides high-level convenience functions, common workflows, and integration utilities
that simplify interaction between the core modules (models, data, configs, utils, analyze).
This module serves as a user-friendly API layer for researchers and practitioners.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from pathlib import Path

# Import all core modules
from utils import (
    set_global_random_seeds,
    setup_experiment_logging,
    create_experiment_directory_structure,
)


# ==============================================================================
# Configuration and Setup Shortcuts
# ==============================================================================


def setup_experiment_environment(
    experiment_name: str, base_dir: Path = Path("./results"), seed: int = 42, device: str = "auto"
) -> Dict[str, Any]:
    """
    Setup complete experiment environment with logging, directories, and seeds.

    Args:
        experiment_name: Name of experiment for organization
        base_dir: Base directory for results
        seed: Random seed for reproducibility
        device: Target device ("auto", "cpu", "cuda")

    Returns:
        Dictionary containing setup information (paths, device, logger)
    """
    # Set random seeds for reproducibility
    set_global_random_seeds(seed)

    # Determine device
    if device == "auto":
        actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        actual_device = torch.device(device)

    experiment_dir = base_dir / experiment_name  # results/abs1


    # Create directory structure
    output_dirs = create_experiment_directory_structure(base_dir, experiment_name)

    # Setup logging
    logger = setup_experiment_logging(
        output_dir=output_dirs.get("logs", experiment_dir), experiment_name=experiment_name, verbosity="INFO"
    )

    # Log setup information
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Device: {actual_device}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Output directory: {experiment_dir}")

    return {
        "experiment_name": experiment_name,
        "device": actual_device,
        "seed": seed,
        "output_dirs": output_dirs,
        "logger": logger,
        "base_dir": base_dir,
    }


