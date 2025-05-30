# utils.py - General Utility Functions for PSL Experiments

"""
Comprehensive utility library supporting all aspects of the Prototype Surface Learning (PSL)
experimentation framework. This module provides foundation services used across all other
components including file I/O, logging, mathematical computations, and reproducibility management.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict
import random
import os
import random
import numpy as np
import torch


# ==============================================================================
# Logging System
# ==============================================================================


def setup_experiment_logging(output_dir: Path, experiment_name: str, verbosity: str = "INFO") -> logging.Logger:
    """
    Configure structured logging for experiment execution.

    Args:
        output_dir: Directory for log files
        experiment_name: Name of experiment for log file naming
        verbosity: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(getattr(logging, verbosity.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler - detailed logging
    log_file = output_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler - user-specified verbosity
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, verbosity.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Log the setup
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Console verbosity: {verbosity}")

    return logger


# ==============================================================================
# Random Seed Management
# ==============================================================================

# Add to utils.py


def set_global_random_seeds(seed: int) -> None:
    """
    Set random seeds for torch, numpy, random, and Python hash seed for reproducibility.

    Args:
        seed: Random seed value
    """

    # Python's built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Determinism and performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash-based randomness
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# Directory and Path Management
# ==============================================================================


def create_experiment_directory_structure(base_path: Path, experiment_name: str) -> Dict[str, Path]:
    """
    Create standard directory structure for experiment results.

    Args:
        base_path: Base directory for all experiments
        experiment_name: Name of specific experiment

    Returns:
        Dictionary mapping directory types to paths
    """
    # Create main experiment directory
    exp_dir = base_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    dirs = {
        "experiment": exp_dir,
        #    "models": exp_dir / "models",
        "logs": exp_dir / "logs",
        #    "plots": exp_dir / "plots",
        #    "analysis": exp_dir / "analysis",
        "runs": exp_dir / "runs",
        #    "configs": exp_dir / "configs",
        #    "checkpoints": exp_dir / "checkpoints"
    }

    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs
