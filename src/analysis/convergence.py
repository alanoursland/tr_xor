import glob
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import argparse
import sys

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pathlib import Path
from sklearn.base import clone

"""
Convergence Prediction Framework

This module implements a machine learning framework for predicting neural network 
convergence time based on current parameter states and final converged states.

GOAL:
Given:
- Current parameters at any point during training
- Final converged parameters (known target state)
Predict:
- Number of epochs needed to reach convergence from current state

APPROACH:
The framework exploits the Markov property of vanilla SGD optimization, where the
current parameter state contains all information needed to predict future convergence.
Each epoch in a training trace becomes a data point:
- Input: (current_params, final_params) 
- Target: epochs_remaining_to_convergence

EXPERIMENTAL DESIGN:
- Uses vanilla SGD (no momentum/optimizer state)
- Parameter state has Markov property 
- Each training epoch provides one prediction sample
- Multiple models compete to learn the state→time relationship

KEY FEATURES:
- Distance metrics: L1, L2, cosine, parameter-wise differences
- Feature engineering: ratios, log transforms, interactions
- Multiple predictive models: linear, polynomial, kernel, neural nets, ensembles
- Statistical validation: bootstrap confidence intervals
- Designed to scale from single traces to large multi-run datasets

This framework helps answer: "How much longer will this model take to converge?"
"""

# ============================================================================
# DATA STRUCTURES (Keep existing dataclasses and enums)
# ============================================================================

class DistanceMetric(Enum):
    L1 = "l1"
    L2 = "l2"
    COSINE = "cosine"
    CHEBYSHEV = "chebyshev"
    PARAMETER_WISE = "parameter_wise"


class RegressionType(Enum):
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    LOG_LINEAR = "log_linear"
    EXPONENTIAL = "exponential"
    KERNEL = "kernel"

@dataclass(frozen=True)
class PathRegistry:
    """
    Canonical place for all on-disk locations that belong to one experiment.
    Keeps path logic in one place so the rest of the code can stay clean.
    """
    experiment: str                              # e.g. "my_run_2025_07_04"
    root: Path = Path("results")                 # default root matches current layout

    # -------- frequently-used subfolders --------
    @property
    def traces(self) -> Path:                    # .../results/<experiment>/parameter_traces
        return self.root / self.experiment / "parameter_traces"

    @property
    def analysis(self) -> Path:                  # .../results/<experiment>/convergence_analysis
        return self.root / self.experiment / "convergence_analysis"

    # -------- helpers --------
    def ensure_dirs(self) -> None:
        """
        Create the writable sub-directories (currently just analysis/)
        so later code can fail fast if the disk is missing / read-only.
        """
        self.analysis.mkdir(parents=True, exist_ok=True)


@dataclass
class ConvergenceData:
    """Container for experiment data"""

    initial_params: torch.Tensor  # Shape: (n_experiments, n_parameters)
    final_params: torch.Tensor  # Shape: (n_experiments, n_parameters)
    epochs_to_converge: torch.Tensor  # Shape: (n_experiments,)
    metadata: Optional[Dict] = None


@dataclass
class PredictionResults:
    """Results from a prediction model"""

    model_type: str
    predictions: torch.Tensor
    actual: torch.Tensor
    r2_score: float
    mse: float
    mae: float
    feature_importance: Optional[torch.Tensor] = None
    model_params: Optional[Dict] = None

# ============================================================================
# MAIN CONVERGENCE PREDICTOR CLASS
# ============================================================================

class ConvergencePredictor:
    """Main API for convergence prediction analysis"""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data: Optional[ConvergenceData] = None
        self.features: Optional[torch.Tensor] = None
        self.results: Dict[str, PredictionResults] = {}
        self.models: Dict[str, Any] = {}

    def load_data(
        self,
        initial_params: Union[torch.Tensor, List],
        final_params: Union[torch.Tensor, List],
        epochs: Union[torch.Tensor, List],
    ) -> None:
        """Load and prepare data for analysis"""
        # Convert to tensors if needed
        if not isinstance(initial_params, torch.Tensor):
            initial_params = torch.tensor(initial_params, dtype=torch.float32)
        if not isinstance(final_params, torch.Tensor):
            final_params = torch.tensor(final_params, dtype=torch.float32)
        if not isinstance(epochs, torch.Tensor):
            epochs = torch.tensor(epochs, dtype=torch.float32)

        # Move to device
        self.data = ConvergenceData(
            initial_params=initial_params.to(self.device),
            final_params=final_params.to(self.device),
            epochs_to_converge=epochs.to(self.device),
        )


    def compute_features(
            self,
            distance_metrics: List[DistanceMetric] = None,
            include_raw: bool = True,
            include_ratios: bool = True,
            include_log: bool = True,
            include_interactions: bool = True,
        ) -> Tuple[torch.Tensor, List[str]]:
            """
            Computes predictive features by delegating to the FeatureEngineering class.
            This method now also populates self.feature_names.
            """
            if self.data is None:
                raise ValueError("No data loaded. Call load_data first.")

            if distance_metrics is None:
                distance_metrics = [
                    DistanceMetric.L1,
                    DistanceMetric.L2,
                    DistanceMetric.PARAMETER_WISE,
                ]
            
            # Delegate the complex work to the new, centralized method
            self.features, self.feature_names = FeatureEngineering.create_all_features(
                initial_params=self.data.initial_params,
                final_params=self.data.final_params,
                distance_metrics=distance_metrics,
                include_raw=include_raw,
                include_ratios=include_ratios,
                include_log=include_log,
                include_interactions=include_interactions,
            )

            return self.features, self.feature_names


    def _polynomial_features(self, X: torch.Tensor, degree: int) -> torch.Tensor:
        """Generate polynomial features"""
        features = [X]
        for d in range(2, degree + 1):
            features.append(X**d)
        return torch.cat(features, dim=1)

    def fit_regression(
        self,
        regression_type: RegressionType,
        degree: int = 2,
        regularization: float = 1e-4,
    ) -> PredictionResults:
        """Fit a regression model without cross-validation."""
        if self.features is None:
            raise ValueError("No features computed. Call compute_features first.")

        X = self.features
        y = self.data.epochs_to_converge
        model_name = regression_type.value
        y_transformed = y

        # --- Model and Data Preparation ---
        if regression_type == RegressionType.LINEAR:
            model = nn.Linear(X.shape[1], 1).to(self.device)
        elif regression_type == RegressionType.POLYNOMIAL:
            X = self._polynomial_features(X, degree)
            model = nn.Linear(X.shape[1], 1).to(self.device)
            model_name = f"poly_{degree}"
        elif regression_type == RegressionType.LOG_LINEAR:
            y_transformed = torch.log(y + 1)
            model = nn.Linear(X.shape[1], 1).to(self.device)
        elif regression_type == RegressionType.EXPONENTIAL:
            y_transformed = torch.log(y + 1e-8)
            model = nn.Linear(X.shape[1], 1).to(self.device)
        else:
            raise NotImplementedError(f"Regression type {regression_type} not implemented")

        # --- Final Model Training ---
        final_model = self._train_pytorch_model(
            model=model,
            X_train=X,
            y_train=y_transformed,
            epochs=1000,
            weight_decay=regularization,
            early_stopping=True
        )

        # --- Evaluation ---
        with torch.no_grad():
            final_predictions = final_model(X).squeeze()
            if regression_type in [RegressionType.LOG_LINEAR, RegressionType.EXPONENTIAL]:
                final_predictions = torch.exp(final_predictions)

            mse = F.mse_loss(final_predictions, y).item()
            mae = F.l1_loss(final_predictions, y).item()
            r2 = 1 - (torch.sum((y - final_predictions) ** 2) / torch.sum((y - y.mean()) ** 2)).item()

        self.models[model_name] = final_model
        result = PredictionResults(
            model_type=model_name,
            predictions=final_predictions,
            actual=y,
            r2_score=r2,
            mse=mse,
            mae=mae,
            model_params={"regularization": regularization},
        )
        self.results[model_name] = result
        return result

    def fit_kernel_regression(
        self, X: torch.Tensor, y: torch.Tensor, kernel: str, regularization: float
    ) -> PredictionResults:
        """Kernel regression implementation with improved numerical stability"""
        n_samples = X.shape[0]

        # Add minimum regularization to ensure numerical stability
        min_regularization = 1e-6
        regularization = max(regularization, min_regularization)

        # Compute kernel matrix
        if kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            # Use median heuristic for gamma
            # We have to do this on the CPU because of memory
            X_cpu = X.cpu()
            dists_cpu = torch.cdist(X_cpu, X_cpu, p=2)
            non_zero_dists_cpu = dists_cpu[dists_cpu > 0]

            # Avoid division by zero in gamma calculation
            if len(non_zero_dists_cpu) > 0:
                median_dist = torch.median(non_zero_dists_cpu)
                torch.clip(median_dist, min=1e-6)
                gamma = 1.0 / (2.0 * median_dist**2)
            else:
                # Fallback gamma if all distances are zero
                gamma = 1.0

            K_cpu = torch.exp(-gamma * dists_cpu ** 2)
            K = K_cpu.to(self.device)
        elif kernel == "linear":
            K = torch.mm(X, X.t())
        elif kernel == "polynomial":
            K = (1 + torch.mm(X, X.t())) ** 3
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Add regularization to diagonal (ridge regression in kernel space)
        K_reg = K + regularization * torch.eye(n_samples, device=self.device)

        # Try to solve the system K * alpha = y
        try:
            # First try direct solve
            alpha = torch.linalg.solve(K_reg, y.unsqueeze(1)).squeeze()
        except torch._C._LinAlgError:
            # If direct solve fails, try with increased regularization
            print(f"Kernel matrix singular with regularization={regularization}, increasing to {regularization * 100}")
            K_reg = K + (regularization * 100) * torch.eye(n_samples, device=self.device)

            try:
                alpha = torch.linalg.solve(K_reg, y.unsqueeze(1)).squeeze()
            except torch._C._LinAlgError:
                # If still failing, use pseudoinverse as last resort
                print("Using pseudoinverse for kernel regression")
                alpha = torch.linalg.pinv(K_reg) @ y.unsqueeze(1)
                alpha = alpha.squeeze()

        # Predictions are K * alpha
        predictions = torch.mv(K, alpha)

        # Calculate metrics
        mse = F.mse_loss(predictions, y).item()
        mae = F.l1_loss(predictions, y).item()

        # Handle potential numerical issues in R2 calculation
        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot).item()
        else:
            r2 = 0.0  # If no variance in y, R2 is undefined

        # Store kernel info for future predictions
        kernel_info = {
            "X_train": X,
            "alpha": alpha,
            "kernel": kernel,
            "gamma": gamma if kernel == "rbf" else None,
            "regularization_used": regularization,
        }

        model_name = f"kernel_{kernel}"
        self.models[model_name] = kernel_info

        result = PredictionResults(
            model_type=model_name,
            predictions=predictions,
            actual=y,
            r2_score=r2,
            mse=mse,
            mae=mae,
            model_params={"kernel": kernel, "regularization": regularization},
        )

        self.results[model_name] = result
        return result

    def fit_ensemble(
        self, n_estimators: int = 100, max_depth: Optional[int] = None, method: str = "random_forest"
    ) -> PredictionResults:
        """Fit ensemble methods (RF, GBM) - GPU accelerated where possible"""
        if self.features is None:
            raise ValueError("No features computed. Call compute_features first.")

        # Move to CPU for sklearn
        X_cpu = self.features.cpu().numpy()
        y_cpu = self.data.epochs_to_converge.cpu().numpy()

        if method == "random_forest":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif method == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth if max_depth else 3, random_state=42
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Fit model
        model.fit(X_cpu, y_cpu)
        predictions = model.predict(X_cpu)

        # Convert back to torch
        predictions_torch = torch.tensor(predictions, device=self.device, dtype=torch.float32)
        y_torch = self.data.epochs_to_converge

        # Calculate metrics
        mse = F.mse_loss(predictions_torch, y_torch).item()
        mae = F.l1_loss(predictions_torch, y_torch).item()
        r2 = 1 - (torch.sum((y_torch - predictions_torch) ** 2) / torch.sum((y_torch - y_torch.mean()) ** 2)).item()

        # Feature importance
        feature_importance = torch.tensor(model.feature_importances_, device=self.device, dtype=torch.float32)

        model_name = f"{method}_{n_estimators}"
        self.models[model_name] = model

        result = PredictionResults(
            model_type=model_name,
            predictions=predictions_torch,
            actual=y_torch,
            r2_score=r2,
            mse=mse,
            mae=mae,
            feature_importance=feature_importance,
            model_params={"n_estimators": n_estimators, "max_depth": max_depth},
        )

        self.results[model_name] = result
        return result

    def fit_neural_network(
            self,
            hidden_layers: List[int],
            activation: str = "relu",
            epochs: int = 1000,
            learning_rate: float = 0.001,
            early_stopping: bool = True,
        ) -> PredictionResults:
            """Fit a neural network predictor by delegating to the unified trainer."""
            if self.features is None:
                raise ValueError("No features computed. Call compute_features first.")

            X = self.features
            y = self.data.epochs_to_converge

            # --- Model Definition ---
            layers = []
            input_dim = X.shape[1]
            activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
            
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activations[activation]())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            model = nn.Sequential(*layers).to(self.device)

            # --- Model Training ---
            # Delegate all training logic to the helper method
            model = self._train_pytorch_model(
                model=model,
                X_train=X,
                y_train=y,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping=early_stopping
            )
            
            # --- Evaluation ---
            with torch.no_grad():
                final_predictions = model(X).squeeze()
                mse = F.mse_loss(final_predictions, y).item()
                mae = F.l1_loss(final_predictions, y).item()
                r2 = 1 - (torch.sum((y - final_predictions) ** 2) / torch.sum((y - y.mean()) ** 2)).item()

            model_name = f"nn_{len(hidden_layers)}layer"
            self.models[model_name] = model

            result = PredictionResults(
                model_type=model_name,
                predictions=final_predictions,
                actual=y,
                r2_score=r2,
                mse=mse,
                mae=mae,
                model_params={"hidden_layers": hidden_layers, "activation": activation},
            )
            self.results[model_name] = result
            return result

    def evaluate_on_test(
        self,
        initial_test: torch.Tensor,
        final_test:   torch.Tensor,
        epochs_test:  torch.Tensor,
        *,
        distance_metrics     = None,
        include_raw          = True,
        include_ratios       = True,
        include_log          = True,
        include_interactions = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate every fitted model on a held-out test set.
        Returns: {model_name: {"r2":…, "mse":…, "mae":…}, …}
        """
        if self.features is None:
            raise ValueError("Train the models first so feature config is fixed.")

        # --- 1. build the test feature matrix (same recipe as training) ----
        X_test, _ = FeatureEngineering.create_all_features(
            initial_test.to(self.device),
            final_test.to(self.device),
            distance_metrics or [DistanceMetric.L1,
                                 DistanceMetric.L2,
                                 DistanceMetric.PARAMETER_WISE],
            include_raw, include_ratios, include_log, include_interactions,
        )
        y_test = epochs_test.to(self.device)

        # --- 2. score each fitted model -----------------------------------
        metrics = {}
        for name, model in self.models.items():

            # -- get predictions ------------------------------------------
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    pred = model(X_test).squeeze()

            elif isinstance(model, dict) and "kernel" in model:
                # kernel regression dict
                X_train = model["X_train"]
                alpha   = model["alpha"]
                ktype   = model["kernel"]
                gamma   = model.get("gamma")

                if ktype == "rbf":
                    K = torch.exp(-gamma * torch.cdist(
                        X_test, X_train, p=2) ** 2)
                elif ktype == "linear":
                    K = torch.mm(X_test, X_train.t())
                elif ktype == "polynomial":
                    K = (1 + torch.mm(X_test, X_train.t())) ** 3
                pred = torch.mv(K, alpha)

            else:  # scikit-learn estimator
                pred = torch.tensor(
                    model.predict(X_test.cpu().numpy()),
                    device=self.device, dtype=torch.float32
                )

            # -- metrics ---------------------------------------------------
            mse = F.mse_loss(pred, y_test).item()
            mae = F.l1_loss(pred, y_test).item()
            r2  = 1 - ((y_test - pred).pow(2).sum() /
                       (y_test - y_test.mean()).pow(2).sum()).item()

            metrics[name] = {"r2": r2, "mse": mse, "mae": mae}

        return metrics

    def compare_models(self, metric: str = "r2", *, use_test: bool = False) -> Dict[str, float]:
        """Return {model_name: score}.  
        metric ∈ {"r2", "mse", "mae"}.  
        If use_test=True, pull from result.test_metrics[metric]."""
        comparison = {}
        for model_name, result in self.results.items():

            # Pick either training or test value --------------------------
            if use_test and hasattr(result, "test_metrics"):
                score = result.test_metrics.get(metric)
            else:   # fall back to training metrics
                if   metric == "r2":  score = result.r2_score
                elif metric == "mse": score = result.mse
                elif metric == "mae": score = result.mae
                else:                 continue  # unknown metric
            # -------------------------------------------------------------

            if score is not None:
                comparison[model_name] = score

        return comparison

    def statistical_tests(self, model1: str, model2: str, test: str = "likelihood_ratio") -> Dict[str, float]:
        """Perform statistical tests between models"""
        if model1 not in self.results or model2 not in self.results:
            raise ValueError("Both models must be fitted first")

        result1 = self.results[model1]
        result2 = self.results[model2]

        y_true = result1.actual
        pred1 = result1.predictions
        pred2 = result2.predictions

        if test == "likelihood_ratio":
            # Assuming Gaussian errors
            n = len(y_true)
            rss1 = torch.sum((y_true - pred1) ** 2)
            rss2 = torch.sum((y_true - pred2) ** 2)

            # Log likelihood ratio
            lr_stat = n * torch.log(rss1 / rss2)

            # Degrees of freedom difference (simplified)
            df = 1  # Assuming models differ by 1 parameter

            # p-value from chi-squared distribution
            from scipy import stats

            p_value = 1 - stats.chi2.cdf(lr_stat.cpu().numpy(), df)

            return {
                "test": "likelihood_ratio",
                "statistic": lr_stat.item(),
                "p_value": p_value,
                "model1_rss": rss1.item(),
                "model2_rss": rss2.item(),
            }

        elif test == "paired_t":
            # Paired t-test on squared errors
            errors1 = (y_true - pred1) ** 2
            errors2 = (y_true - pred2) ** 2
            diff = errors1 - errors2

            mean_diff = torch.mean(diff)
            std_diff = torch.std(diff, unbiased=True)
            n = len(diff)

            t_stat = mean_diff / (std_diff / torch.sqrt(torch.tensor(n, dtype=torch.float32)))

            from scipy import stats

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat.cpu().numpy()), n - 1))

            return {
                "test": "paired_t",
                "statistic": t_stat.item(),
                "p_value": p_value,
                "mean_difference": mean_diff.item(),
            }

        elif test == "diebold_mariano":
            # Diebold-Mariano test for forecast accuracy
            e1 = y_true - pred1
            e2 = y_true - pred2
            d = e1**2 - e2**2  # Squared loss differential

            mean_d = torch.mean(d)
            var_d = torch.var(d, unbiased=True)
            n = len(d)

            dm_stat = mean_d / torch.sqrt(var_d / n)

            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat.cpu().numpy())))

            return {
                "test": "diebold_mariano",
                "statistic": dm_stat.item(),
                "p_value": p_value,
                "mean_loss_diff": mean_d.item(),
            }

        else:
            raise ValueError(f"Unknown test: {test}")

    def feature_selection(
        self, method: str = "lasso", n_features: Optional[int] = None, alpha: float = 1.0
    ) -> torch.Tensor:
        """Select most important features"""
        if self.features is None:
            raise ValueError("No features computed. Call compute_features first.")

        X = self.features
        y = self.data.epochs_to_converge

        if method == "lasso":
            # L1-regularized linear regression
            model = nn.Linear(X.shape[1], 1).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Train with L1 regularization
            for epoch in range(1000):
                optimizer.zero_grad()
                predictions = model(X).squeeze()
                mse_loss = F.mse_loss(predictions, y)
                l1_loss = alpha * torch.sum(torch.abs(model.weight))
                loss = mse_loss + l1_loss
                loss.backward()
                optimizer.step()

            # Get feature importance (absolute weights)
            feature_importance = torch.abs(model.weight.squeeze())

        elif method == "variance":
            # Select features with highest variance
            feature_variance = torch.var(X, dim=0)
            feature_importance = feature_variance

        elif method == "correlation":
            # Select features most correlated with target
            correlations = []
            for i in range(X.shape[1]):
                corr = torch.abs(torch.corrcoef(torch.stack([X[:, i], y]))[0, 1])
                correlations.append(corr)
            feature_importance = torch.tensor(correlations)

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Get top n features
        if n_features is not None:
            top_indices = torch.topk(feature_importance, n_features).indices
        else:
            # Return all features sorted by importance
            top_indices = torch.argsort(feature_importance, descending=True)

        return top_indices

    def bootstrap_confidence(
        self, model: str, n_bootstrap: int = 1000, confidence: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """Bootstrap confidence intervals for predictions - GPU optimized"""
        if model not in self.models:
            raise ValueError(f"Model {model} not found. Train it first.")

        if self.features is None:
            raise ValueError("No features computed.")

        X = self.features
        y = self.data.epochs_to_converge
        n_samples = X.shape[0]

        bootstrap_predictions = []

        # Get the model
        fitted_model = self.models[model]

        # Check if this is a sklearn model (CPU-only)
        if not isinstance(fitted_model, (nn.Module, dict)):
            print(f"  Note: {model} is a sklearn model, running bootstrap on CPU")
            print(f"  This will be slower than PyTorch models. Consider using a PyTorch model for faster bootstrap.")

            # Reduce bootstrap samples for sklearn models to save time
            n_bootstrap = min(n_bootstrap, 50)
            print(f"  Reducing bootstrap samples to {n_bootstrap} for sklearn model")

        # Perform bootstrap
        for i in range(n_bootstrap):
            if i % 20 == 0:  # Progress indicator
                print(f"    Bootstrap iteration {i+1}/{n_bootstrap}")

            # Sample with replacement
            indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
            X_boot = X[indices]
            y_boot = y[indices]

            # Handle different model types
            if isinstance(fitted_model, nn.Module):
                # PyTorch models - recreate and train
                boot_model = self._recreate_model(fitted_model)
                
                # Train the bootstrap model
                boot_model = self._train_pytorch_model(
                    model=boot_model,
                    X_train=X_boot,
                    y_train=y_boot,
                    epochs=50,  # Reduced for bootstrap speed
                    learning_rate=0.01,
                    early_stopping=False  # No early stopping for bootstrap
                )

                # Get predictions on full dataset
                boot_model.eval()
                with torch.no_grad():
                    boot_pred = boot_model(X).squeeze()
                    bootstrap_predictions.append(boot_pred)

            elif isinstance(fitted_model, dict):
                # Kernel regression - special handling
                if "kernel" in fitted_model:
                    kernel_type = fitted_model["kernel"]
                    regularization = fitted_model.get("regularization_used", 1e-4)

                    # GPU kernel computation
                    if kernel_type == "rbf":
                        dists = torch.cdist(X_boot, X_boot, p=2)
                        non_zero_dists = dists[dists > 0]
                        if len(non_zero_dists) > 0:
                            median_dist = torch.median(non_zero_dists)
                            torch.clip(median_dist, min=1e-6)
                            gamma = 1.0 / (2.0 * median_dist**2)
                        else:
                            gamma = fitted_model.get("gamma", 1.0)
                        K_boot = torch.exp(-gamma * dists**2)
                    elif kernel_type == "linear":
                        K_boot = torch.mm(X_boot, X_boot.t())
                    elif kernel_type == "polynomial":
                        K_boot = (1 + torch.mm(X_boot, X_boot.t())) ** 3

                    # Add regularization
                    K_boot_reg = K_boot + regularization * torch.eye(X_boot.shape[0], device=self.device)

                    # GPU solve
                    alpha_boot = torch.linalg.solve(K_boot_reg, y_boot.unsqueeze(1)).squeeze()

                    # Predict on full dataset (GPU)
                    if kernel_type == "rbf":
                        K_full = torch.exp(-gamma * torch.cdist(X, X_boot, p=2) ** 2)
                    elif kernel_type == "linear":
                        K_full = torch.mm(X, X_boot.t())
                    elif kernel_type == "polynomial":
                        K_full = (1 + torch.mm(X, X_boot.t())) ** 3

                    boot_pred = torch.mv(K_full, alpha_boot)
                    bootstrap_predictions.append(boot_pred)
            else:
                # Sklearn models
                boot_model = self._recreate_model(fitted_model)
                X_boot_cpu = X_boot.cpu().numpy()
                y_boot_cpu = y_boot.cpu().numpy()
                X_cpu = X.cpu().numpy()

                boot_model.fit(X_boot_cpu, y_boot_cpu)
                boot_pred = boot_model.predict(X_cpu)
                boot_pred_tensor = torch.tensor(boot_pred, device=self.device, dtype=torch.float32)
                bootstrap_predictions.append(boot_pred_tensor)

        if len(bootstrap_predictions) == 0:
            raise ValueError(f"Could not perform bootstrap for model type: {type(fitted_model)}")

        print(f"    Completed {len(bootstrap_predictions)} successful bootstrap iterations")

        # Calculate confidence intervals on GPU
        bootstrap_predictions = torch.stack(bootstrap_predictions)
        alpha = (1 - confidence) / 2
        lower_percentile = int(alpha * len(bootstrap_predictions))
        upper_percentile = int((1 - alpha) * len(bootstrap_predictions))

        sorted_preds, _ = torch.sort(bootstrap_predictions, dim=0)

        return {
            "lower": sorted_preds[lower_percentile],
            "upper": sorted_preds[upper_percentile],
            "mean": bootstrap_predictions.mean(dim=0),
            "std": bootstrap_predictions.std(dim=0),
            "n_successful_bootstraps": len(bootstrap_predictions),
        }

    def visualize_results(self, save_path: Path) -> None:
        """Generate diagnostic plots using stored feature names."""

        if not self.results:
            print("No results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Model comparison
        ax = axes[0, 0]
        models = list(self.results.keys())
        r2_scores = [self.results[m].r2_score for m in models]
        ax.bar(models, r2_scores)
        ax.set_title("Model R² Scores")
        ax.set_ylabel("R²")
        ax.tick_params(axis="x", rotation=45)

        # 2. Best model predictions vs actual
        best_model = max(self.results.items(), key=lambda x: x[1].r2_score)[0]
        best_result = self.results[best_model]

        ax = axes[0, 1]
        actual = best_result.actual.cpu().numpy()
        predicted = best_result.predictions.cpu().numpy()
        ax.scatter(actual, predicted, alpha=0.5)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--")
        ax.set_xlabel("Actual Epochs")
        ax.set_ylabel("Predicted Epochs")
        ax.set_title(f"Best Model ({best_model}) Predictions")

        # 3. Residuals
        ax = axes[1, 0]
        residuals = actual - predicted
        ax.scatter(predicted, residuals, alpha=0.5)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")

        # 4. Feature importance (if available)
        ax = axes[1, 1]
        ax.set_title("Top 10 Feature Importances")
        best_model_name, best_result = max(self.results.items(), key=lambda item: item[1].r2_score)

        if best_result.feature_importance is not None:
            importances = best_result.feature_importance.cpu().numpy()
            
            indices = np.argsort(importances)[-10:] # Get top 10
            top_features = [self.feature_names[i] for i in indices]
            top_importances = importances[indices]

            ax.barh(top_features, top_importances)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title(f"Top 10 Features for {best_model_name}")
        else:
            ax.text(0.5, 0.5, "Feature importance not available\nfor the best model.", 
                    ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def interpret_best_model(self, model_name: str, output_dir: Path) -> None:
        """
        Provides interpretability for the best model and saves artifacts to disk.

        Args:
            model_name: The name of the model to interpret.
            feature_names: A mapping of feature indices to human-readable names.
            output_dir: The directory Path object where plots should be saved.
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Cannot perform interpretation.")
            return

        print("\n" + "="*80)
        print(f"INTERPRETATION FOR BEST MODEL: {model_name}")
        print("="*80)

        model = self.models[model_name]
        X_cpu = self.features.cpu().numpy()
        y_cpu = self.data.epochs_to_converge.cpu().numpy()

        # 1. Ranked Feature Importance (if available)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]

            print("\n--- Top 10 Most Important Features ---")
            for i in range(min(10, len(importances))):
                idx = sorted_indices[i]
                feature_name = self.feature_names[idx]
                print(f"{i+1}. {feature_name}: {importances[idx]:.4f}")

        # 2. Partial Dependence Plots for top features
        print("\nGenerating Partial Dependence Plots for top 2 features...")
        fig, ax = plt.subplots(figsize=(12, 6))
        top_two_indices = sorted_indices[:2]
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_cpu,
            features=top_two_indices,
            feature_names=self.feature_names,
            ax=ax
        )
        plt.suptitle(f"Partial Dependence Plots for {model_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Create interpretation output directory
        fig_path = output_dir / "partial_dependence_plots.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"  Plots saved to {fig_path}")


        # 3. SHAP Analysis (for tree models)
        # if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        #     print("\nCalculating SHAP values for model interpretation...")
            
        #     # ================== NEW: SAMPLING CODE ==================
        #     # To prevent long runtimes, we'll use a random sample for SHAP analysis.
        #     X_sample = shap.sample(X_cpu, 2000)
        #     # ========================================================
            
        #     explainer = shap.TreeExplainer(model)
        #     shap_values = explainer.shap_values(X_sample) # Use the sample here

        #     # Generate and save SHAP summary plot
        #     shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        #     fig_path = output_dir / "shap_summary_plot.png"
        #     plt.title(f"SHAP Feature Impact for {model_name}")
        #     plt.savefig(fig_path, bbox_inches='tight')
        #     plt.close()
        #     print(f"  SHAP summary plot saved to {fig_path}")

        # 4. Surrogate Decision Tree Model
        # ... (This section is unchanged) ...
        print("\n--- Approximating with a Simple Decision Tree (Surrogate Model) ---")
        surrogate_model = DecisionTreeRegressor(max_depth=4, random_state=42)
        
        black_box_predictions = model.predict(X_cpu)
        surrogate_model.fit(X_cpu, black_box_predictions)
        
        surrogate_r2 = surrogate_model.score(X_cpu, black_box_predictions)
        print(f"Simple tree R² (approximating Random Forest): {surrogate_r2:.4f}")

        if surrogate_r2 > 0.75:
            print("The simple tree is a good approximation. Here are its rules:")
            tree_rules = export_text(surrogate_model, feature_names=self.feature_names, max_depth=3)
            print(tree_rules)
        else:
            print("The surrogate model is not a close enough approximation to display simple rules.")

    def _train_pytorch_model(
            self,
            model: nn.Module,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            epochs: int = 1000,
            learning_rate: float = 0.01,
            weight_decay: float = 0.0,
            early_stopping: bool = True,
        ) -> nn.Module:
            """
            A centralized training loop for any PyTorch nn.Module.

            Args:
                model: The PyTorch model to be trained.
                X_train: Training features.
                y_train: Training targets.
                epochs: Maximum number of training epochs.
                learning_rate: The learning rate for the Adam optimizer.
                weight_decay: L2 regularization term for the optimizer.
                early_stopping: Flag to enable early stopping based on training loss.

            Returns:
                The trained PyTorch model.
            """
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            criterion = nn.MSELoss()

            best_loss = float("inf")
            patience = 50
            patience_counter = 0
            best_state = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                predictions = model(X_train).squeeze()
                loss = criterion(predictions, y_train)
                loss.backward()
                optimizer.step()

                if early_stopping:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                        # Store the state of the best model found so far
                        best_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Restore the best model state if early stopping is triggered
                            if best_state:
                                model.load_state_dict(best_state)
                            # print(f"  Early stopping at epoch {epoch+1}") # Optional: for debugging
                            break
            
            # Ensure the model is in evaluation mode after training
            model.eval()
            return model

    def _recreate_model(self, original_model):
        """
        Create a fresh copy of a model for bootstrap sampling.
        
        Args:
            original_model: The fitted model to recreate (can be nn.Module, dict, or sklearn model)
            
        Returns:
            A new unfitted model with the same architecture
        """
        if isinstance(original_model, nn.Linear):
            # Simple linear model
            new_model = nn.Linear(original_model.in_features, original_model.out_features).to(self.device)
            new_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
            return new_model
            
        elif isinstance(original_model, nn.Sequential):
            # Neural network with multiple layers
            layers = []
            for layer in original_model:
                if isinstance(layer, nn.Linear):
                    layers.append(nn.Linear(layer.in_features, layer.out_features))
                elif isinstance(layer, nn.ReLU):
                    layers.append(nn.ReLU())
                elif isinstance(layer, nn.Tanh):
                    layers.append(nn.Tanh())
                elif isinstance(layer, nn.Sigmoid):
                    layers.append(nn.Sigmoid())
            new_model = nn.Sequential(*layers).to(self.device)
            new_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
            return new_model
            
        elif isinstance(original_model, dict):
            # Kernel regression - return the dict as-is (will be handled differently)
            return original_model
            
        else:
            # Sklearn models
            from sklearn.base import clone
            return clone(original_model)

    def _extract_detailed_linear_equation(self) -> str:
        """
        Extracts the full linear equation with proper feature names using
        the internally stored model and feature_names list.
        """
        model_name = "linear" # Assuming the simple linear model is named 'linear'
        if model_name not in self.models:
            return "Linear model not found or has not been trained."
        
        linear_model = self.models[model_name]
        if not isinstance(linear_model, torch.nn.Linear):
             return f"Model '{model_name}' is not a torch.nn.Linear module."

        weights = linear_model.weight.data.cpu().numpy().flatten()
        bias = linear_model.bias.data.cpu().numpy()[0]
        
        # Use self.feature_names as the single source of truth
        if len(weights) != len(self.feature_names):
            return "Mismatch between number of weights and feature names."
            
        equation_parts = [f"{bias:.4f}"]
        for i, weight in enumerate(weights):
            if abs(weight) > 1e-6:  # Only include significant terms
                feature_name = self.feature_names[i]
                sign = "+" if weight >= 0 else "-"
                equation_parts.append(f" {sign} {abs(weight):.4f} * {feature_name}")
        
        return "epochs_remaining = " + "".join(equation_parts)


# ============================================================================
# FEATURE ENGINEERING UTILITIES
# ============================================================================
class FeatureEngineering:
    """
    GPU-accelerated feature engineering utilities.

    This class centralizes all logic for creating predictive features from
    raw model parameters. The main entry point `create_all_features` returns
    both the feature tensor and a list of human-readable names, ensuring
    that data and its description are always synchronized.
    """

    @staticmethod
    def create_all_features(
        initial_params: torch.Tensor,
        final_params: torch.Tensor,
        distance_metrics: List[DistanceMetric],
        include_raw: bool,
        include_ratios: bool,
        include_log: bool,
        include_interactions: bool,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generates a comprehensive feature set and their corresponding names.

        Args:
            initial_params: Tensor of current model parameters.
            final_params: Tensor of final (converged) model parameters.
            distance_metrics: A list of distance metrics to compute.
            include_raw: Flag to include raw parameter differences.
            include_ratios: Flag to include parameter ratios.
            include_log: Flag to include log-transformed differences.
            include_interactions: Flag to include pairwise interaction terms.

        Returns:
            A tuple containing:
            - The concatenated feature tensor.
            - A list of strings with the name for each feature column.
        """
        all_features: List[torch.Tensor] = []
        all_feature_names: List[str] = []
        n_params = initial_params.shape[1]

        # --- Distance Metrics ---
        if distance_metrics:
            dist_tensor, dist_names = FeatureEngineering._compute_aggregate_distances(
                initial_params, final_params, distance_metrics
            )
            all_features.append(dist_tensor)
            all_feature_names.extend(dist_names)

        # --- Raw Parameter Differences ---
        if include_raw:
            diff = initial_params - final_params
            all_features.append(diff)
            all_feature_names.extend([f"raw_diff_p{i}" for i in range(n_params)])

        # --- Parameter-wise Absolute Differences ---
        if DistanceMetric.PARAMETER_WISE in distance_metrics:
            abs_diff = torch.abs(initial_params - final_params)
            all_features.append(abs_diff)
            all_feature_names.extend([f"abs_diff_p{i}" for i in range(n_params)])

        # --- Ratios ---
        if include_ratios:
            # Add a small epsilon to avoid division by zero
            ratios = initial_params / (final_params + 1e-8)
            all_features.append(ratios)
            all_feature_names.extend([f"ratio_p{i}" for i in range(n_params)])

        # --- Log Transforms ---
        if include_log:
            # Use absolute difference for log transform to avoid log(negative)
            log_diff = torch.log(torch.abs(initial_params - final_params) + 1e-8)
            all_features.append(log_diff)
            all_feature_names.extend([f"log_abs_diff_p{i}" for i in range(n_params)])

        # --- Interaction Terms (Pairwise Products of Differences) ---
        if include_interactions and n_params > 1:
            diff = initial_params - final_params
            interactions = []
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    interactions.append((diff[:, i] * diff[:, j]).unsqueeze(1))
                    all_feature_names.append(f"interaction_p{i}_p{j}")
            if interactions:
                all_features.append(torch.cat(interactions, dim=1))

        if not all_features:
            raise ValueError("No features were selected for computation.")

        # Concatenate all features into a single tensor
        final_feature_tensor = torch.cat(all_features, dim=1)

        return final_feature_tensor, all_feature_names

    @staticmethod
    def _compute_aggregate_distances(
        initial: torch.Tensor, final: torch.Tensor, metrics: List[DistanceMetric]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Helper to compute distance metrics that result in a single value per sample.
        Excludes PARAMETER_WISE as it's not an aggregate metric.
        """
        distances: List[torch.Tensor] = []
        names: List[str] = []

        for metric in metrics:
            if metric == DistanceMetric.L1:
                dist = torch.sum(torch.abs(initial - final), dim=1, keepdim=True)
                distances.append(dist)
                names.append("L1_distance")
            elif metric == DistanceMetric.L2:
                dist = torch.norm(initial - final, p=2, dim=1, keepdim=True)
                distances.append(dist)
                names.append("L2_distance")
            elif metric == DistanceMetric.COSINE:
                cos_sim = F.cosine_similarity(initial, final, dim=1, eps=1e-8)
                dist = (1 - cos_sim).unsqueeze(1) # Convert similarity to distance
                distances.append(dist)
                names.append("cosine_distance")
            elif metric == DistanceMetric.CHEBYSHEV:
                dist = torch.max(torch.abs(initial - final), dim=1, keepdim=True)[0]
                distances.append(dist)
                names.append("chebyshev_distance")
            elif metric == DistanceMetric.PARAMETER_WISE:
                # This is handled separately as it doesn't aggregate
                continue

        if not distances:
            return torch.empty(initial.shape[0], 0, device=initial.device), []

        return torch.cat(distances, dim=1), names

    @staticmethod
    def compute_distances(initial: torch.Tensor, final: torch.Tensor, metrics: List[DistanceMetric]) -> torch.Tensor:
        """Compute multiple distance metrics efficiently"""
        distances = []

        for metric in metrics:
            if metric == DistanceMetric.L1:
                dist = torch.sum(torch.abs(initial - final), dim=1, keepdim=True)
                distances.append(dist)

            elif metric == DistanceMetric.L2:
                dist = torch.norm(initial - final, p=2, dim=1, keepdim=True)
                distances.append(dist)

            elif metric == DistanceMetric.COSINE:
                # Cosine distance = 1 - cosine similarity
                cos_sim = F.cosine_similarity(initial, final, dim=1, eps=1e-8)
                dist = (1 - cos_sim).unsqueeze(1)
                distances.append(dist)

            elif metric == DistanceMetric.CHEBYSHEV:
                # Maximum absolute difference
                dist = torch.max(torch.abs(initial - final), dim=1, keepdim=True)[0]
                distances.append(dist)

            elif metric == DistanceMetric.PARAMETER_WISE:
                # Keep all parameter-wise distances
                dist = torch.abs(initial - final)
                distances.append(dist)

        return torch.cat(distances, dim=1)

    @staticmethod
    def polynomial_features(X: torch.Tensor, degree: int) -> torch.Tensor:
        """Generate polynomial features on GPU"""
        n_samples, n_features = X.shape
        features = [torch.ones(n_samples, 1, device=X.device, dtype=X.dtype), X]

        # Generate polynomial terms
        for d in range(2, degree + 1):
            # Generate all combinations of degree d
            if d == 2:
                # Quadratic terms including interactions
                quad_features = []
                for i in range(n_features):
                    for j in range(i, n_features):
                        if i == j:
                            quad_features.append((X[:, i] ** 2).unsqueeze(1))
                        else:
                            quad_features.append((X[:, i] * X[:, j]).unsqueeze(1))
                features.append(torch.cat(quad_features, dim=1))
            else:
                # Higher degree terms (no interactions for simplicity)
                features.append(X**d)

        return torch.cat(features, dim=1)

    @staticmethod
    def log_transform(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Log transformation with numerical stability"""
        # Handle negative values by taking log of absolute value
        # and preserving sign information
        sign_X = torch.sign(X)
        log_abs_X = torch.log(torch.abs(X) + epsilon)

        # Combine sign and log magnitude
        # Option 1: Multiply (preserves sign in output)
        transformed = sign_X * log_abs_X

        # Option 2: Stack sign as separate features
        # transformed = torch.cat([log_abs_X, sign_X], dim=1)

        return transformed


# ============================================================================
# MODEL FACTORY
# ============================================================================
class ModelFactory:
    """Factory for creating GPU-accelerated models"""

    @staticmethod
    def create_linear_model(input_dim: int, regularization: str = "none", alpha: float = 1.0) -> torch.nn.Module:
        """Create linear regression model with optional regularization"""

        class RegularizedLinear(nn.Module):
            def __init__(self, input_dim, regularization, alpha):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
                self.regularization = regularization
                self.alpha = alpha

            def forward(self, x):
                return self.linear(x)

            def regularization_loss(self):
                if self.regularization == "l1":
                    return self.alpha * torch.sum(torch.abs(self.linear.weight))
                elif self.regularization == "l2":
                    return self.alpha * torch.sum(self.linear.weight**2)
                else:
                    return 0.0

        return RegularizedLinear(input_dim, regularization, alpha)

    @staticmethod
    def create_polynomial_model(input_dim: int, degree: int, regularization: str = "none") -> torch.nn.Module:
        """Create polynomial regression model"""
        # First expand features to polynomial
        # This returns a module that does polynomial expansion + linear regression

        class PolynomialRegression(nn.Module):
            def __init__(self, input_dim, degree, regularization):
                super().__init__()
                # Calculate output dimension after polynomial expansion
                # For simplicity, just powers up to degree (no interactions)
                self.degree = degree
                poly_dim = input_dim * degree + 1  # +1 for bias term

                self.linear = nn.Linear(poly_dim, 1)
                self.regularization = regularization

            def forward(self, x):
                # Generate polynomial features
                features = [torch.ones(x.shape[0], 1, device=x.device)]
                features.append(x)

                for d in range(2, self.degree + 1):
                    features.append(x**d)

                poly_x = torch.cat(features, dim=1)
                return self.linear(poly_x)

            def regularization_loss(self):
                if self.regularization == "l1":
                    return torch.sum(torch.abs(self.linear.weight))
                elif self.regularization == "l2":
                    return torch.sum(self.linear.weight**2)
                else:
                    return 0.0

        return PolynomialRegression(input_dim, degree, regularization)

    @staticmethod
    def create_kernel_model(kernel: str = "rbf", gamma: float = 1.0) -> Callable:
        """Create kernel regression model (returns a function)"""

        def rbf_kernel(X1, X2, gamma=gamma):
            """RBF kernel function"""
            dists = torch.cdist(X1, X2, p=2)
            return torch.exp(-gamma * dists**2)

        def linear_kernel(X1, X2):
            """Linear kernel function"""
            return torch.mm(X1, X2.t())

        def polynomial_kernel(X1, X2, degree=3):
            """Polynomial kernel function"""
            return (1 + torch.mm(X1, X2.t())) ** degree

        if kernel == "rbf":
            return rbf_kernel
        elif kernel == "linear":
            return linear_kernel
        elif kernel == "polynomial":
            return polynomial_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")


# Usage example (not implementation):
"""
predictor = ConvergencePredictor(device='cuda')
predictor.load_data(initial_params, final_params, epochs)
predictor.compute_features(
    distance_metrics=[DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.COSINE],
    include_interactions=True
)

# Try multiple models
linear_results = predictor.fit_regression(RegressionType.LINEAR)
poly_results = predictor.fit_regression(RegressionType.POLYNOMIAL, degree=3)
nn_results = predictor.fit_neural_network([64, 32, 16])

# Compare and test
comparison = predictor.compare_models()
predictor.statistical_tests('linear', 'polynomial')

# Get best features
important_features = predictor.feature_selection(method='lasso', n_features=5)
"""

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_multiple_traces(registry: PathRegistry) -> Dict:
    """
    Analyze multiple training traces to build robust convergence predictors.
    
    Uses each epoch from each trace as a data point: from epoch i parameters, 
    predict how many epochs remain to reach final convergence.

    Args:
        trace_files: List of paths to JSON trace files

    Returns:
        Dictionary with analysis results
    """

    trace_dir = registry.traces
    output_dir = registry.analysis

    # Find all trace files
    trace_files = sorted(registry.traces.glob('*.json'))

    print(f"Found {len(trace_files)} trace files")


    print(f"=== Analyzing {len(trace_files)} training traces ===")
    
    # Initialize predictor
    print("Initializing ConvergencePredictor...")
    predictor = ConvergencePredictor(device="cuda")

    # Collect data from all traces
    all_initial_params   = []
    all_final_params     = []
    all_epochs_remaining = []
    all_trace_ids        = []          # <-- NEW: keep run/trace labels
    trace_metadata = []
    total_epochs_processed = 0

    # Keep track of skipped files
    skipped_files = 0

    print("Processing trace files...")
    
    # Process each trace file
    for i, trace_file in enumerate(trace_files):
        print(f"\nProcessing trace {i+1}/{len(trace_files)}: {os.path.basename(trace_file)}")
        
        # Load trace data
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        accuracy = 0.0
        # Prioritize checking the metadata object for final accuracy
        if 'metadata' in trace_data and 'final_accuracy' in trace_data['metadata']:
            accuracy = trace_data['metadata']['final_accuracy']
        # Fallback to checking the final epoch if not in metadata
        elif trace_data.get("epochs"):
                final_epoch = trace_data["epochs"][-1]
                if 'accuracy' in final_epoch:
                    accuracy = final_epoch['accuracy']
                elif 'metrics' in final_epoch and 'accuracy' in final_epoch['metrics']:
                    accuracy = final_epoch['metrics']['accuracy']

        # Skip the trace if final accuracy is not 100%
        if accuracy < 1.0:
            print(f"  Skipping trace: Final accuracy is {accuracy:.2%}, not 100%.")
            skipped_files += 1
            continue
        
        print(f"  Trace passed: Final accuracy is {accuracy}. Proceeding with analysis.")
        
        # Extract data from this trace
        epochs = trace_data["epochs"]
        final_epoch = epochs[-1]
        print(f"  Found {len(epochs)} epochs in this trace")

        # Convert final parameters to list
        final_params = []
        for key, value in final_epoch["parameters"].items():
            if isinstance(value, list):
                flat_params = np.array(value).flatten()
                final_params.extend(flat_params)

        # Create data points from each epoch in this trace
        trace_initial_params = []
        trace_final_params = []
        trace_epochs_remaining = []

        # Use each epoch as a potential starting point (excluding final epoch)
        for j, epoch in enumerate(epochs[:-1]):
            # Get parameters at epoch j
            epoch_params = []
            for key, value in epoch["parameters"].items():
                if isinstance(value, list):
                    flat_params = np.array(value).flatten()
                    epoch_params.extend(flat_params)

            trace_initial_params.append(epoch_params)
            trace_final_params.append(final_params)
            # Epochs remaining from this point
            epochs_remaining = len(epochs) - 1 - j
            trace_epochs_remaining.append(epochs_remaining)

        # Add this trace's data to the global collection
        all_initial_params.extend(trace_initial_params)
        all_final_params.extend(trace_final_params)
        all_epochs_remaining.extend(trace_epochs_remaining)
        all_trace_ids.extend([i] * len(trace_initial_params))
        
        # Store metadata
        trace_metadata.append(trace_data["metadata"])
        
        total_epochs_processed += len(epochs)
        print(f"  Added {len(trace_initial_params)} data points from this trace")
        print(f"  Epochs remaining range: {min(trace_epochs_remaining)} to {max(trace_epochs_remaining)}")

    if not all_initial_params:
        raise ValueError("No valid data found in any trace files")

    # -------------------------------------------------------------------
    # --- HOLD-OUT WHOLE RUNS FOR AN HONEST TEST SET --------------------
    # -------------------------------------------------------------------

    # 1. assign a trace-id to every sample (already built earlier)
    #    → all_trace_ids  == one int per row in all_initial_params
    unique_runs  = np.unique(all_trace_ids)
    test_runs    = np.random.choice(
        unique_runs,
        size=max(1, int(0.20 * len(unique_runs))),   # 20 % of runs
        replace=False
    )

    test_mask  = np.isin(all_trace_ids, test_runs)
    train_mask = ~test_mask

    # 2. slice each list with the masks
    def mask_list(lst, mask):
        return [row for row, keep in zip(lst, mask) if keep]

    train_initial = mask_list(all_initial_params,  train_mask)
    train_final   = mask_list(all_final_params,    train_mask)
    train_epochs  = mask_list(all_epochs_remaining,train_mask)

    test_initial  = mask_list(all_initial_params,  test_mask)
    test_final    = mask_list(all_final_params,    test_mask)
    test_epochs   = mask_list(all_epochs_remaining,test_mask)

    print(f"\nTrain runs : {train_mask.sum()} samples  "
        f"from {len(unique_runs) - len(test_runs)} traces")
    print(  f"Test  runs : {test_mask.sum()} samples  "
        f"from {len(test_runs)} traces")

    print(f"\n=== Data collection summary ===")
    print(f"Total traces processed: {len(trace_metadata)}")
    print(f"Total epochs processed: {total_epochs_processed}")
    print(f"Total data points created: {len(all_initial_params)}")
    print(f"Overall epochs remaining range: {min(all_epochs_remaining)} to {max(all_epochs_remaining)}")

    print("\nConverting *training* set to tensors...")
    # Convert the training lists into GPU/CPU tensors
    initial_params_tensor = torch.tensor(np.array(train_initial), dtype=torch.float32)
    final_params_tensor   = torch.tensor(np.array(train_final),   dtype=torch.float32)
    epochs_tensor         = torch.tensor(train_epochs,            dtype=torch.float32)

    print(f"  Train tensor shapes: "
        f"initial={initial_params_tensor.shape}, "
        f"final={final_params_tensor.shape}, "
        f"epochs={epochs_tensor.shape}")

    print("\nConverting *test* set to tensors...")
    # Convert the held-out lists into tensors (stay on CPU/GPU via default device logic)
    test_initial_tensor = torch.tensor(np.array(test_initial), dtype=torch.float32)
    test_final_tensor   = torch.tensor(np.array(test_final),  dtype=torch.float32)
    test_epochs_tensor  = torch.tensor(test_epochs,           dtype=torch.float32)

    print(f"  Test tensor shapes: "
        f"initial={test_initial_tensor.shape}, "
        f"final={test_final_tensor.shape}, "
        f"epochs={test_epochs_tensor.shape}")

    print("Loading data into predictor...")
    # Load data into predictor
    predictor.load_data(initial_params_tensor, final_params_tensor, epochs_tensor)

    print("Computing features...")
    # Compute features with multiple distance metrics
    features = predictor.compute_features(
        distance_metrics=[DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.PARAMETER_WISE],
        include_raw=True,
        include_ratios=True,
        include_log=True,
        include_interactions=True,
    )

    print(f"Features computed: {len(features[1])}")
    print(f"  Feature types: L1, L2, Cosine, Parameter-wise distances + raw diffs + ratios + log + interactions")

    # Fit various models
    results = {}

    # 1. Simple linear regression
    # print("1. Training linear regression...")
    # results["linear"] = predictor.fit_regression(RegressionType.LINEAR)
    # print(f"   Linear Train R²: {results['linear'].r2_score:.4f}")

    # 2. Polynomial regression x^2
    # print("2. Training polynomial regression (degree=2)...")
    # results["poly2"] = predictor.fit_regression(RegressionType.POLYNOMIAL, degree=2)
    # print(f"   Poly2 Train R²: {results['poly2'].r2_score:.4f}")

    # 3. Polynomial regression x^3
    # print("3. Training polynomial regression (degree=3)...")
    # results["poly3"] = predictor.fit_regression(RegressionType.POLYNOMIAL, degree=3)
    # print(f"   Poly3 Train R²: {results['poly3'].r2_score:.4f}")

    # 4. Log-linear (for exponential decay)
    # print("4. Training log-linear regression...")
    # results["log_linear"] = predictor.fit_regression(RegressionType.LOG_LINEAR)
    # print(f"   Log-linear Train R²: {results['log_linear'].r2_score:.4f}")

    # 5. Exponential regression
    # print("5. Training exponential regression...")
    # results["exponential"] = predictor.fit_regression(RegressionType.EXPONENTIAL)
    # print(f"   Exponential Train R²: {results['exponential'].r2_score:.4f}")

    # 6. Kernel regression
    # print("6. Training RBF kernel regression...")
    # results["kernel_rbf"] = predictor.fit_kernel_regression(predictor.features, predictor.data.epochs_to_converge, kernel="rbf", regularization=1e-4)
    # print(f"   Kernel RBF Train R²: {results['kernel_rbf'].r2_score:.4f}")

    # print("7. Training linear kernel regression...")
    # results["kernel_linear"] = predictor.fit_kernel_regression(predictor.features, predictor.data.epochs_to_converge, kernel="linear", regularization=1e-4)
    # print(f"   Kernel Linear Train R²: {results['kernel_linear'].r2_score:.4f}")

    # 8. Neural networks (with more data, we can try bigger networks)
    # print("8. Training neural network (small)...")
    # results["neural_net_small"] = predictor.fit_neural_network(
    #     hidden_layers=[32, 16], activation="relu", epochs=500, early_stopping=True
    # )
    # print(f"   Neural net (small) Train R²: {results['neural_net_small'].r2_score:.4f}")

    # if len(all_epochs_remaining) >= 100:
    #     print("9. Training neural network (large)...")
    #     results["neural_net_large"] = predictor.fit_neural_network(
    #         hidden_layers=[64, 32, 16], activation="relu", epochs=1000, early_stopping=True
    #     )
    #     print(f"   Neural net (large) Train R²: {results['neural_net_large'].r2_score:.4f}")
    # else:
    #     print("9. Skipping large neural network (insufficient data)")

    # 10. Ensemble methods
    print("10. Training random forest...")
    results["random_forest"] = predictor.fit_ensemble(n_estimators=100, method="random_forest")
    print(f"    Random forest Train R²: {results['random_forest'].r2_score:.4f}")

    # print("11. Training gradient boosting...")
    # results["gradient_boosting"] = predictor.fit_ensemble(n_estimators=100, method="gradient_boosting")
    # print(f"    Gradient boosting Train R²: {results['gradient_boosting'].r2_score:.4f}")

    # ------------------------------------------------------------------
    # === EVALUATE EVERY MODEL ON THE HELD-OUT TEST RUNS ===============
    # ------------------------------------------------------------------
    print("\n=== Generalisation to unseen runs (test set) ===")

    # 1) build test feature matrix (identical options to training call)
    test_features, _ = FeatureEngineering.create_all_features(
        initial_params=test_initial_tensor.to(predictor.device),
        final_params=test_final_tensor.to(predictor.device),
        distance_metrics=[DistanceMetric.L1, DistanceMetric.L2,
                        DistanceMetric.COSINE, DistanceMetric.PARAMETER_WISE],
        include_raw=True,
        include_ratios=True,
        include_log=True,
        include_interactions=True,
    )

    y_test = test_epochs_tensor.to(predictor.device)

    test_metrics = {}   # {model_name: {"r2":…, "mse":…, "mae":…}}

    for name, model in predictor.models.items():
        # --- get predictions ------------------------------------------
        # If the model is polynomial, expand test features too
        X_test = test_features
        if "poly_" in name:
            degree = int(name.split("_")[1])
            X_test = predictor._polynomial_features(X_test, degree)

        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                pred = model(X_test).squeeze()

        elif isinstance(model, dict) and "kernel" in model:  # kernel regression
            X_train = model["X_train"]
            alpha   = model["alpha"]
            ktype   = model["kernel"]
            gamma   = model.get("gamma")

            if ktype == "rbf":
                K = torch.exp(-gamma * torch.cdist(test_features, X_train, p=2) ** 2)
            elif ktype == "linear":
                K = torch.mm(test_features, X_train.t())
            elif ktype == "polynomial":
                K = (1 + torch.mm(test_features, X_train.t())) ** 3
            pred = torch.mv(K, alpha)

        else:  # scikit-learn estimators
            pred = torch.tensor(
                model.predict(test_features.cpu().numpy()),
                device=predictor.device, dtype=torch.float32
            )

        # --- metrics ---------------------------------------------------
        mse = torch.nn.functional.mse_loss(pred, y_test).item()
        mae = torch.nn.functional.l1_loss(pred, y_test).item()
        r2  = 1 - ((y_test - pred).pow(2).sum() /
                (y_test - y_test.mean()).pow(2).sum()).item()

        test_metrics[name] = {"r2": r2, "mse": mse, "mae": mae}
        if name in predictor.results:
            predictor.results[name].__dict__["test_metrics"] = test_metrics[name]
        # print(f"{name:20s}  R²={r2:.4f}   MSE={mse:.3f}   MAE={mae:.3f}")

    for name, metrics in sorted(test_metrics.items(), key=lambda x: x[1]["r2"], reverse=True):
        print(f"{name:20s}  R²={metrics['r2']:.4f}   MSE={metrics['mse']:.3f}   MAE={metrics['mae']:.3f}")

    # ------------------------------------------------------------------

    print("\n=== Model comparison and analysis ===")

    # Compare models
    model_comparison = predictor.compare_models(metric="r2", use_test=True)
    print("Model R² scores:")
    for model, r2 in sorted(model_comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {r2:.4f}")

    print("\nPerforming feature selection...")
    # Feature selection - find most important features
    important_features = predictor.feature_selection(method="lasso", n_features=min(10, len(features)), alpha=0.1)
    print(f"  Selected {len(important_features)} most important features")

    # Also try correlation-based feature selection
    corr_features = predictor.feature_selection(method="correlation", n_features=min(10, len(features)))
    print(f"  Correlation-based selection: {len(corr_features)} features")

    # Statistical tests between top models
    sorted_models = sorted(model_comparison.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_models) >= 2:
        best_model = sorted_models[0][0]
        second_best = sorted_models[1][0]
        print(f"\nRunning statistical test between {best_model} and {second_best}...")
        stat_test = predictor.statistical_tests(best_model, second_best)
        print(f"  Test result: p-value = {stat_test.get('p_value', 'N/A')}")
    else:
        stat_test = None

    # Bootstrap confidence intervals for best model
    best_model_name = sorted_models[0][0]
    # print(f"\nComputing bootstrap confidence intervals for {best_model_name}...")
    # confidence_intervals = predictor.bootstrap_confidence(model=best_model_name, n_bootstrap=200, confidence=0.95)
    # print(f"  Bootstrap completed with {confidence_intervals.get('n_successful_bootstraps', 'unknown')} successful samples")
    
    # Call the new interpretation function for the best model
    predictor.interpret_best_model(best_model_name, output_dir=registry.analysis)

    # Map internal model names back to results keys
    model_name_mapping = {
        "gradient_boosting_100": "gradient_boosting",
        "random_forest_100": "random_forest", 
        "random_forest_50": "random_forest",
        "nn_2layer": "neural_net_small",
        "nn_3layer": "neural_net_large",
        "poly_2": "poly2",
        "poly_3": "poly3"
    }
    
    # Get the correct key for the results dictionary
    results_key = model_name_mapping.get(best_model_name, best_model_name)

    print("\nAnalyzing dataset characteristics...")
    # Analyze the combined dataset
    first_trace_initial = torch.tensor(all_initial_params[0], dtype=torch.float32)
    first_trace_final = torch.tensor(all_final_params[0], dtype=torch.float32)
    avg_distance_l2 = torch.norm(initial_params_tensor - final_params_tensor, p=2, dim=1).mean().item()
    avg_distance_l1 = torch.norm(initial_params_tensor - final_params_tensor, p=1, dim=1).mean().item()
    
    print(f"  Average L2 distance (current→final): {avg_distance_l2:.4f}")
    print(f"  Average L1 distance (current→final): {avg_distance_l1:.4f}")
    print(f"  Parameter count: {initial_params_tensor.shape[1]}")

    print("\n=== Multi-trace analysis complete ===")
    print(f"Processed {len(trace_metadata)} traces with {len(all_initial_params)} total data points")
    print(f"Best model: {best_model_name} (R² = {results[results_key].r2_score:.4f})")

    print("\nPreparing summary...")
    # Prepare comprehensive summary
    summary = {
        "dataset_metadata": {
            "total_traces": len(trace_metadata),
            "total_data_points": len(all_initial_params),
            "total_epochs_processed": total_epochs_processed,
            "parameter_count": initial_params_tensor.shape[1],
            "feature_count": len(features),
            "epochs_range": {
                "min": min(all_epochs_remaining),
                "max": max(all_epochs_remaining),
                "mean": np.mean(all_epochs_remaining)
            }
        },
        "trace_metadata": trace_metadata,
        "dataset_analysis": {
            "avg_l2_distance": avg_distance_l2,
            "avg_l1_distance": avg_distance_l1,
            "distance_statistics": {
                "l2_std": torch.norm(initial_params_tensor - final_params_tensor, p=2, dim=1).std().item(),
                "l1_std": torch.norm(initial_params_tensor - final_params_tensor, p=1, dim=1).std().item(),
            }
        },
        "model_comparison": model_comparison,
        "best_model": {
            "name": best_model_name,
            "r2_score": results[results_key].r2_score,
            "mse": results[results_key].mse,
            "mae": results[results_key].mae,
        },
        "feature_analysis": {
            "lasso_selected_features": (
                important_features.tolist() if torch.is_tensor(important_features) else important_features
            ),
            "correlation_selected_features": (
                corr_features.tolist() if torch.is_tensor(corr_features) else corr_features
            ),
            "feature_descriptions": "Indices of most predictive features for convergence time",
        },
        "statistical_significance": stat_test,
        # "prediction_confidence": {
        #     "lower_bound": confidence_intervals["lower"].tolist() if "lower" in confidence_intervals else None,
        #     "upper_bound": confidence_intervals["upper"].tolist() if "upper" in confidence_intervals else None,
        #     "bootstrap_samples": confidence_intervals.get('n_successful_bootstraps', 0)
        # },
        "convergence_patterns": {
            "traces_analyzed": len(trace_metadata),
            "average_convergence_epochs": np.mean([meta["total_epochs"] for meta in trace_metadata]),
            "convergence_variance": np.var([meta["total_epochs"] for meta in trace_metadata])
        }
    }

    print("Generating visualization plots...")
    # Generate plots

    os.makedirs(output_dir, exist_ok=True)
    fig_path = registry.analysis / "multi_trace_convergence_analysis.png"
    predictor.visualize_results(save_path=fig_path)
    print(f"  Plots saved to: {fig_path}")

    print("\n=== Multi-trace analysis complete ===")
    print(f"Processed {len(trace_metadata)} traces with {len(all_initial_params)} total data points")
    print(f"Best model: {best_model_name} (R² = {results[results_key].r2_score:.4f})")
    
    return summary


def analyze_convergence_pattern(epochs: List[Dict]) -> Dict:
    """
    Analyze the convergence pattern from epoch data.
    """
    losses = [e["loss"] for e in epochs]

    # Calculate convergence metrics
    loss_ratios = [losses[i + 1] / losses[i] for i in range(len(losses) - 1)]
    avg_ratio = np.mean(loss_ratios)

    # Find point where loss < 0.01 (1% of initial)
    convergence_threshold = losses[0] * 0.01
    epochs_to_1_percent = next((i for i, loss in enumerate(losses) if loss < convergence_threshold), len(losses))

    # Estimate if convergence is exponential
    log_losses = np.log(losses)
    epochs_array = np.arange(len(losses))
    correlation = np.corrcoef(epochs_array, log_losses)[0, 1]

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "average_loss_ratio": avg_ratio,
        "epochs_to_1_percent": epochs_to_1_percent,
        "log_linear_correlation": correlation,
        "is_exponential": abs(correlation) > 0.95,
    }


def document_feature_equations(predictor: ConvergencePredictor) -> Dict[str, List[str]]:
    """
    Document the mathematical equations for all synthetic features.
    """
    # Get the parameter count from the data
    n_params = predictor.data.initial_params.shape[1]
    param_names = [f"p{i}" for i in range(n_params)]  # p0, p1, p2, etc.
    
    feature_equations = {
        "distance_metrics": [],
        "raw_differences": [],
        "ratios": [], 
        "log_transforms": [],
        "interactions": []
    }
    
    # Distance metrics
    feature_equations["distance_metrics"] = [
        f"L1_distance = |{param_names[0]}_init - {param_names[0]}_final| + |{param_names[1]}_init - {param_names[1]}_final| + |{param_names[2]}_init - {param_names[2]}_final|",
        f"L2_distance = sqrt(({param_names[0]}_init - {param_names[0]}_final)² + ({param_names[1]}_init - {param_names[1]}_final)² + ({param_names[2]}_init - {param_names[2]}_final)²)",
        f"cosine_distance = 1 - (initial_params · final_params) / (||initial_params|| × ||final_params||)",
        f"chebyshev_distance = max(|{param_names[0]}_init - {param_names[0]}_final|, |{param_names[1]}_init - {param_names[1]}_final|, |{param_names[2]}_init - {param_names[2]}_final|)"
    ]
    
    # Raw parameter differences
    for i in range(n_params):
        feature_equations["raw_differences"].append(
            f"diff_{param_names[i]} = {param_names[i]}_init - {param_names[i]}_final"
        )
    
    # Ratios
    for i in range(n_params):
        feature_equations["ratios"].append(
            f"ratio_{param_names[i]} = {param_names[i]}_init / ({param_names[i]}_final + 1e-8)"
        )
    
    # Log transforms
    for i in range(n_params):
        feature_equations["log_transforms"].append(
            f"log_diff_{param_names[i]} = log(|{param_names[i]}_init - {param_names[i]}_final| + 1e-8)"
        )
    
    # Interaction terms (pairwise products)
    for i in range(n_params):
        for j in range(i + 1, n_params):
            feature_equations["interactions"].append(
                f"interaction_{param_names[i]}_{param_names[j]} = diff_{param_names[i]} × diff_{param_names[j]}"
            )
    
    return feature_equations


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

# ============================================================================
# INTERPRETATION AND FEATURE MAPPING
# ============================================================================


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()

    registry = PathRegistry(args.experiment_name)
    # fail-fast if someone mis-specifies or the directory isn’t there
    if not registry.traces.exists() or not registry.traces.is_dir():
        raise FileNotFoundError(f"Trace directory {registry.traces!r} not found or not a directory")

    registry.ensure_dirs()          # make sure analysis/ exists

    analysis_results = analyze_multiple_traces(registry)

    summary_path = registry.analysis / "analysis_summary.json"
    print(f"\nSaving detailed summary to {summary_path}...")
    with open(summary_path, 'w') as f:
        # Custom converter to handle numpy/torch objects in the summary
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(analysis_results, f, indent=4, cls=NpEncoder)
    print("  Summary saved.")

    # Print summary
    print("Convergence Analysis Results")
    print("=" * 50)
    print(f"Total traces analyzed: {analysis_results['dataset_metadata']['total_traces']}")
    print(f"Total data points: {analysis_results['dataset_metadata']['total_data_points']}")
    print(f"Parameter count: {analysis_results['dataset_metadata']['parameter_count']}")
    print(f"Average L2 distance (current→final): {analysis_results['dataset_analysis']['avg_l2_distance']:.4f}")
    print(f"Average L1 distance (current→final): {analysis_results['dataset_analysis']['avg_l1_distance']:.4f}")
    print(f"\nBest predictive model: {analysis_results['best_model']['name']}")
    print(f"R² score: {analysis_results['best_model']['r2_score']:.4f}")
    print(f"MSE: {analysis_results['best_model']['mse']:.4f}")
    print(f"MAE: {analysis_results['best_model']['mae']:.4f}")

    # Show convergence statistics
    conv_patterns = analysis_results['convergence_patterns']
    print(f"\nConvergence statistics:")
    print(f"Average epochs to convergence: {conv_patterns['average_convergence_epochs']:.1f}")
    print(f"Convergence time variance: {conv_patterns['convergence_variance']:.2f}")

    # Show model comparison (top 5)
    print(f"\nTop 5 model performances:")
    model_comp = analysis_results['model_comparison']
    sorted_models = sorted(model_comp.items(), key=lambda x: x[1], reverse=True)
    for i, (model, r2) in enumerate(sorted_models[:5]):
        print(f"  {i+1}. {model}: R² = {r2:.4f}")

    # print("\n" + "="*80)
    # print("MATHEMATICAL EQUATIONS FOR CONVERGENCE PREDICTION")
    # print("="*80)

    # print("\n=== FEATURE ENGINEERING EQUATIONS ===")
    # print("The following synthetic features are computed from raw parameters:")

    # print("\nDistance Metrics:")
    # print("  L1_distance = |p0_init - p0_final| + |p1_init - p1_final| + |p2_init - p2_final|")
    # print("  L2_distance = sqrt((p0_init - p0_final)² + (p1_init - p1_final)² + (p2_init - p2_final)²)")
    # print("  cosine_distance = 1 - (initial_params · final_params) / (||initial_params|| × ||final_params||)")

    # print("\nRaw Parameter Differences:")
    # print("  diff_p0 = p0_init - p0_final")
    # print("  diff_p1 = p1_init - p1_final") 
    # print("  diff_p2 = p2_init - p2_final")

    # print("\nParameter Ratios:")
    # print("  ratio_p0 = p0_init / (p0_final + 1e-8)")
    # print("  ratio_p1 = p1_init / (p1_final + 1e-8)")
    # print("  ratio_p2 = p2_init / (p2_final + 1e-8)")

    # print("\nLog Transforms:")
    # print("  log_diff_p0 = log(|p0_init - p0_final| + 1e-8)")
    # print("  log_diff_p1 = log(|p1_init - p1_final| + 1e-8)")
    # print("  log_diff_p2 = log(|p2_init - p2_final| + 1e-8)")

    # print("\nInteraction Terms:")
    # print("  interaction_p0_p1 = diff_p0 × diff_p1")
    # print("  interaction_p0_p2 = diff_p0 × diff_p2")
    # print("  interaction_p1_p2 = diff_p1 × diff_p2")

    # print("\n=== CONVERGENCE PREDICTION EQUATIONS ===")

    # # Print all model equations with their accuracy
    # model_comparison = analysis_results['model_comparison']
    # sorted_models = sorted(model_comparison.items(), key=lambda x: x[1], reverse=True)

    # print("\nAll Trained Models (ranked by accuracy):")
    # for i, (model_name, r2_score) in enumerate(sorted_models):
    #     print(f"\n{i+1}. {model_name.upper()} (R² = {r2_score:.4f}):")
        
    #     if "linear" in model_name and "log" not in model_name and "kernel" not in model_name:
    #         print("   epochs_remaining = β₀ + β₁×L1_distance + β₂×L2_distance + β₃×cosine_distance + ...")
    #         print("   [Linear combination of all 18 features]")
            
    #     elif "poly" in model_name:
    #         degree = "2" if "poly_2" in model_name else "3"
    #         print(f"   epochs_remaining = polynomial expansion of degree {degree}")
    #         print("   [Includes squared terms, cubic terms, and cross-products of features]")
            
    #     elif "log_linear" in model_name:
    #         print("   log(epochs_remaining) = β₀ + β₁×L1_distance + β₂×L2_distance + ...")
    #         print("   epochs_remaining = exp(β₀ + β₁×L1_distance + β₂×L2_distance + ...)")
            
    #     elif "exponential" in model_name:
    #         print("   epochs_remaining = exp(β₀ + β₁×L1_distance + β₂×L2_distance + ...)")
            
    #     elif "kernel_rbf" in model_name:
    #         print("   epochs_remaining = Σᵢ αᵢ × exp(-γ||x - xᵢ||²)")
    #         print("   [Radial Basis Function kernel with Gaussian similarity]")
            
    #     elif "kernel_linear" in model_name:
    #         print("   epochs_remaining = Σᵢ αᵢ × (x · xᵢ)")
    #         print("   [Linear kernel regression]")
            
    #     elif "neural_net" in model_name or "nn_" in model_name:
    #         if "small" in model_name or "2layer" in model_name:
    #             print("   epochs_remaining = NN([32, 16] hidden units)")
    #         else:
    #             print("   epochs_remaining = NN([64, 32, 16] hidden units)")
    #         print("   [Multi-layer neural network with ReLU activations]")
            
    #     elif "random_forest" in model_name:
    #         print("   epochs_remaining = average of 100 decision trees")
    #         print("   [Ensemble of decision trees with feature bagging]")
            
    #     elif "gradient_boosting" in model_name:
    #         print("   epochs_remaining = Σₜ learning_rate × tree_t(features)")
    #         print("   [Sequential ensemble of 100 boosted decision trees]")
            
    #     else:
    #         print("   [Complex non-linear model]")

    # Show the best performing models summary
    best_model_name = sorted_models[0][0]
    print(f"\n🏆 BEST MODEL: {best_model_name} (R² = {sorted_models[0][1]:.4f})")
    if sorted_models[0][1] >= 0.99:
        print("   This model achieves near-perfect prediction accuracy!")
        print("   The relationship between parameter state and convergence time is highly predictable.")

    # Show simple interpretable alternatives
    # print(f"\n📊 SIMPLE INTERPRETABLE MODELS:")
    # for model_name, r2_score in sorted_models:
    #     if "linear" in model_name and "log" not in model_name and "kernel" not in model_name and r2_score > 0.95:
    #         print(f"   Linear model: R² = {r2_score:.4f} (highly interpretable)")
    #         break

