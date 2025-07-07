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
- Statistical validation: cross-validation, bootstrap confidence intervals
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
    cross_val_scores: Optional[torch.Tensor] = None
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
        custom_features: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        Delegate to FeatureEngineering so all distances, logs, ratios, interactions
        stay in one place.
+        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")

        if distance_metrics is None:
            distance_metrics = [DistanceMetric.L1, DistanceMetric.L2]

        features = []

        # Raw parameter differences
        if include_raw:
            diff = self.data.initial_params - self.data.final_params
            features.append(diff)

        # Distance metrics
        for metric in distance_metrics:
            if metric == DistanceMetric.L1:
                dist = torch.abs(self.data.initial_params - self.data.final_params).sum(dim=1, keepdim=True)
                features.append(dist)
            elif metric == DistanceMetric.L2:
                dist = torch.norm(self.data.initial_params - self.data.final_params, p=2, dim=1, keepdim=True)
                features.append(dist)
            elif metric == DistanceMetric.COSINE:
                cos_sim = F.cosine_similarity(
                    self.data.initial_params, self.data.final_params, dim=1, eps=1e-8
                ).unsqueeze(1)
                features.append(1 - cos_sim)  # Convert to distance
            elif metric == DistanceMetric.CHEBYSHEV:
                dist = torch.abs(self.data.initial_params - self.data.final_params).max(dim=1, keepdim=True)[0]
                features.append(dist)
            elif metric == DistanceMetric.PARAMETER_WISE:
                dist = torch.abs(self.data.initial_params - self.data.final_params)
                features.append(dist)

        # Ratios
        if include_ratios:
            ratios = self.data.initial_params / (self.data.final_params + 1e-8)
            features.append(ratios)

        # Log features
        if include_log:
            log_diff = torch.log(torch.abs(self.data.initial_params - self.data.final_params) + 1e-8)
            features.append(log_diff)

        # Interaction terms (pairwise products)
        if include_interactions and self.data.initial_params.shape[1] > 1:
            diff = self.data.initial_params - self.data.final_params
            n_params = diff.shape[1]
            interactions = []
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    interactions.append((diff[:, i] * diff[:, j]).unsqueeze(1))
            if interactions:
                features.append(torch.cat(interactions, dim=1))

        # Custom features
        if custom_features:
            for func in custom_features:
                custom_feat = func(self.data.initial_params, self.data.final_params)
                if custom_feat.dim() == 1:
                    custom_feat = custom_feat.unsqueeze(1)
                features.append(custom_feat)

        # Concatenate all features
        self.features = torch.cat(features, dim=1)
        return self.features

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
        kernel: str = "rbf",
        regularization: float = 1e-4,  # Changed default from 0.0 to 1e-4
        cv_folds: int = 5,
    ) -> PredictionResults:
        """Fit a regression model with cross-validation - completing missing types"""
        if self.features is None:
            raise ValueError("No features computed. Call compute_features first.")

        X = self.features
        y = self.data.epochs_to_converge

        if regression_type == RegressionType.LINEAR:
            model = nn.Linear(X.shape[1], 1).to(self.device)
            model_name = "linear"
        elif regression_type == RegressionType.POLYNOMIAL:
            X = self._polynomial_features(X, degree)
            model = nn.Linear(X.shape[1], 1).to(self.device)
            model_name = f"poly_{degree}"
        elif regression_type == RegressionType.LOG_LINEAR:
            y = torch.log(y + 1)
            model = nn.Linear(X.shape[1], 1).to(self.device)
            model_name = "log_linear"
        elif regression_type == RegressionType.EXPONENTIAL:
            # For exponential: log(y) = a + bx, so y = exp(a + bx)
            y_transformed = torch.log(y + 1e-8)
            model = nn.Linear(X.shape[1], 1).to(self.device)
            model_name = "exponential"
        elif regression_type == RegressionType.KERNEL:
            # Implement RBF kernel regression
            model_name = f"kernel_{kernel}"
            return self._fit_kernel_regression(X, y, kernel, regularization, cv_folds)
        else:
            raise NotImplementedError(f"Regression type {regression_type} not implemented")

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=regularization)
        criterion = nn.MSELoss()

        # Cross-validation
        if cv_folds > 1:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in kf.split(X.cpu().numpy()):
                # Clone model for CV
                cv_model = type(model)(model.in_features, model.out_features).to(self.device)
                cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=0.01, weight_decay=regularization)

                train_idx = torch.tensor(train_idx, device=self.device)
                val_idx = torch.tensor(val_idx, device=self.device)

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train CV model
                for epoch in range(500):
                    cv_optimizer.zero_grad()
                    if regression_type == RegressionType.EXPONENTIAL:
                        predictions = cv_model(X_train).squeeze()
                        loss = criterion(predictions, y_transformed[train_idx])
                    else:
                        predictions = cv_model(X_train).squeeze()
                        loss = criterion(predictions, y_train)
                    loss.backward()
                    cv_optimizer.step()

                # Validate
                cv_model.eval()
                with torch.no_grad():
                    val_pred = cv_model(X_val).squeeze()
                    if regression_type == RegressionType.EXPONENTIAL:
                        val_pred = torch.exp(val_pred) - 1e-8
                        val_true = torch.exp(y_transformed[val_idx]) - 1e-8
                    else:
                        val_true = y_val

                    val_r2 = (
                        1
                        - (torch.sum((val_true - val_pred) ** 2) / torch.sum((val_true - val_true.mean()) ** 2)).item()
                    )
                    cv_scores.append(val_r2)

        # Train final model on all data
        for epoch in range(1000):
            optimizer.zero_grad()
            if regression_type == RegressionType.EXPONENTIAL:
                predictions = model(X).squeeze()
                loss = criterion(predictions, y_transformed)
            else:
                predictions = model(X).squeeze()
                loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

        # Get final predictions
        model.eval()
        with torch.no_grad():
            final_predictions = model(X).squeeze()

            if regression_type == RegressionType.LOG_LINEAR:
                final_predictions = torch.exp(final_predictions) - 1
                y = torch.exp(y) - 1
            elif regression_type == RegressionType.EXPONENTIAL:
                final_predictions = torch.exp(final_predictions) - 1e-8
                y = self.data.epochs_to_converge  # Original y

            mse = F.mse_loss(final_predictions, y).item()
            mae = F.l1_loss(final_predictions, y).item()
            r2 = 1 - (torch.sum((y - final_predictions) ** 2) / torch.sum((y - y.mean()) ** 2)).item()

        self.models[model_name] = model
        result = PredictionResults(
            model_type=model_name,
            predictions=final_predictions,
            actual=y,
            r2_score=r2,
            mse=mse,
            mae=mae,
            cross_val_scores=torch.tensor(cv_scores) if cv_folds > 1 else None,
            model_params={"regularization": regularization},
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
        """Fit a neural network predictor"""
        if self.features is None:
            raise ValueError("No features computed. Call compute_features first.")

        X = self.features
        y = self.data.epochs_to_converge

        # Build neural network
        layers = []
        input_dim = X.shape[1]

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        model = nn.Sequential(*layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training with early stopping
        best_loss = float("inf")
        patience = 50
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X).squeeze()
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            if early_stopping:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        model.load_state_dict(best_state)
                        break

        # Evaluate
        model.eval()
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
            model_params={"hidden_layers": hidden_layers, "activation": activation, "epochs_trained": epoch + 1},
        )

        self.results[model_name] = result
        return result

    def compare_models(self, models: Optional[List[str]] = None, metric: str = "r2") -> Dict[str, float]:
        """Compare all fitted models by specified metric"""
        if models is None:
            models = list(self.results.keys())

        comparison = {}
        for model_name in models:
            if model_name in self.results:
                result = self.results[model_name]
                if metric == "r2":
                    comparison[model_name] = result.r2_score
                elif metric == "mse":
                    comparison[model_name] = result.mse
                elif metric == "mae":
                    comparison[model_name] = result.mae

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
                # PyTorch models - keep everything on GPU
                if isinstance(fitted_model, nn.Linear):
                    boot_model = nn.Linear(fitted_model.in_features, fitted_model.out_features).to(self.device)
                elif isinstance(fitted_model, nn.Sequential):
                    # Recreate architecture
                    layers = []
                    for layer in fitted_model:
                        if isinstance(layer, nn.Linear):
                            layers.append(nn.Linear(layer.in_features, layer.out_features))
                        elif isinstance(layer, nn.ReLU):
                            layers.append(nn.ReLU())
                        elif isinstance(layer, nn.Tanh):
                            layers.append(nn.Tanh())
                        elif isinstance(layer, nn.Sigmoid):
                            layers.append(nn.Sigmoid())
                    boot_model = nn.Sequential(*layers).to(self.device)

                # Reset parameters
                boot_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

                # Fast GPU training with fewer epochs for bootstrap
                optimizer = torch.optim.Adam(boot_model.parameters(), lr=0.01)
                criterion = nn.MSELoss()

                # Reduced epochs for bootstrap speed
                for _ in range(50):  # Reduced from 100
                    optimizer.zero_grad()
                    pred = boot_model(X_boot).squeeze()
                    loss = criterion(pred, y_boot)
                    loss.backward()
                    optimizer.step()

                # Get predictions on full dataset (GPU)
                boot_model.eval()
                with torch.no_grad():
                    boot_pred = boot_model(X).squeeze()
                    bootstrap_predictions.append(boot_pred)

            elif isinstance(fitted_model, dict):
                # Kernel regression - keep on GPU
                if "kernel" in fitted_model:
                    kernel_type = fitted_model["kernel"]
                    regularization = fitted_model.get("regularization_used", 1e-4)

                    # GPU kernel computation
                    if kernel_type == "rbf":
                        dists = torch.cdist(X_boot, X_boot, p=2)
                        non_zero_dists = dists[dists > 0]
                        if len(non_zero_dists) > 0:
                            median_dist = torch.median(non_zero_dists)
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
                # Sklearn models - must use CPU
                from sklearn.base import clone

                boot_model = clone(fitted_model)
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

    def visualize_results(self, plot_type: str = "all", save_path: Optional[str] = None) -> None:
        """Generate diagnostic plots - simplified for research"""

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
        for model_name, result in self.results.items():
            if result.feature_importance is not None:
                importance = result.feature_importance.cpu().numpy()
                ax.bar(range(len(importance)), importance, alpha=0.5, label=model_name)
                ax.set_title("Feature Importance")
                ax.set_xlabel("Feature Index")
                ax.set_ylabel("Importance")
                ax.legend()
                break

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def _fit_kernel_regression(
        self, X: torch.Tensor, y: torch.Tensor, kernel: str, regularization: float, cv_folds: int
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
            dists = torch.cdist(X, X, p=2)

            # Avoid division by zero in gamma calculation
            non_zero_dists = dists[dists > 0]
            if len(non_zero_dists) > 0:
                median_dist = torch.median(non_zero_dists)
                gamma = 1.0 / (2.0 * median_dist**2)
            else:
                # Fallback gamma if all distances are zero
                gamma = 1.0

            K = torch.exp(-gamma * dists**2)
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

    def interpret_best_model(self, model_name: str, feature_names: Dict[int, str], trace_dir: str) -> None:
            """
            Provides interpretability for the best performing model using SHAP,
            feature importances, and a surrogate decision tree.
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

            # 1. Ranked Feature Importance
            # ... (This section is unchanged) ...
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]

                print("\n--- Top 10 Most Important Features ---")
                for i in range(min(10, len(importances))):
                    idx = sorted_indices[i]
                    feature_name = feature_names.get(idx, f"feature_{idx}")
                    print(f"{i+1}. {feature_name}: {importances[idx]:.4f}")

                # 2. Partial Dependence Plots for top features
                # ... (This section is unchanged) ...
                print("\nGenerating Partial Dependence Plots for top 2 features...")
                fig, ax = plt.subplots(figsize=(12, 6))
                top_two_indices = sorted_indices[:2]
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_cpu,
                    features=top_two_indices,
                    feature_names=[feature_names.get(i, f"f_{i}") for i in range(X_cpu.shape[1])],
                    ax=ax
                )
                plt.suptitle(f"Partial Dependence Plots for {model_name}")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                # Create interpretation output directory
                fig_path = registry.analysis / "partial_dependence_plots.png"
                plt.savefig(fig_path)
                plt.close()
                print(f"  Plots saved to {fig_path}")


            # 3. SHAP Analysis (for tree models)
            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                print("\nCalculating SHAP values for model interpretation...")
                
                # ================== NEW: SAMPLING CODE ==================
                # To prevent long runtimes, we'll use a random sample for SHAP analysis.
                X_sample = shap.sample(X_cpu, 2000)
                # ========================================================
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample) # Use the sample here

                # Generate and save SHAP summary plot
                shap.summary_plot(shap_values, X_sample, feature_names=[feature_names.get(i, f"f_{i}") for i in range(X_cpu.shape[1])], show=False)
                fig_path = registry.analysis / "shap_summary_plot.png"
                plt.title(f"SHAP Feature Impact for {model_name}")
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()
                print(f"  SHAP summary plot saved to {fig_path}")

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
                feature_name_list = [feature_names.get(i, f"feature_{i}") for i in range(X_cpu.shape[1])]
                tree_rules = export_text(surrogate_model, feature_names=feature_name_list, max_depth=3)
                print(tree_rules)
            else:
                print("The surrogate model is not a close enough approximation to display simple rules.")

# ============================================================================
# FEATURE ENGINEERING UTILITIES
# ============================================================================
class FeatureEngineering:
    """GPU-accelerated feature engineering utilities"""

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
linear_results = predictor.fit_regression(RegressionType.LINEAR, cv_folds=10)
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
    all_initial_params = []
    all_final_params = []
    all_epochs_remaining = []
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
        
        # Store metadata
        trace_metadata.append(trace_data["metadata"])
        
        total_epochs_processed += len(epochs)
        print(f"  Added {len(trace_initial_params)} data points from this trace")
        print(f"  Epochs remaining range: {min(trace_epochs_remaining)} to {max(trace_epochs_remaining)}")

    if not all_initial_params:
        raise ValueError("No valid data found in any trace files")

    print(f"\n=== Data collection summary ===")
    print(f"Total traces processed: {len(trace_metadata)}")
    print(f"Total epochs processed: {total_epochs_processed}")
    print(f"Total data points created: {len(all_initial_params)}")
    print(f"Overall epochs remaining range: {min(all_epochs_remaining)} to {max(all_epochs_remaining)}")

    print("\nConverting to tensors...")
    # Convert to tensors
    initial_params_tensor = torch.tensor(np.array(all_initial_params), dtype=torch.float32)
    final_params_tensor = torch.tensor(np.array(all_final_params), dtype=torch.float32)
    epochs_tensor = torch.tensor(all_epochs_remaining, dtype=torch.float32)
    print(f"  Tensor shapes: initial={initial_params_tensor.shape}, final={final_params_tensor.shape}, epochs={epochs_tensor.shape}")

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

    print(f"Features computed: {features.shape}")
    print(f"  Feature types: L1, L2, Cosine, Parameter-wise distances + raw diffs + ratios + log + interactions")

    # Fit various models
    results = {}

    print("\n=== Training prediction models ===")
    cv_folds = min(10, len(all_epochs_remaining) // 5)  # More folds with more data
    print(f"Using {cv_folds}-fold cross-validation")
    
    # 1. Simple linear regression
    print("1. Training linear regression...")
    results["linear"] = predictor.fit_regression(RegressionType.LINEAR, cv_folds=cv_folds)
    print(f"   Linear R²: {results['linear'].r2_score:.4f}")

    # 2. Polynomial regression
    print("2. Training polynomial regression (degree=2)...")
    results["poly2"] = predictor.fit_regression(RegressionType.POLYNOMIAL, degree=2, cv_folds=cv_folds)
    print(f"   Poly2 R²: {results['poly2'].r2_score:.4f}")

    print("3. Training polynomial regression (degree=3)...")
    results["poly3"] = predictor.fit_regression(RegressionType.POLYNOMIAL, degree=3, cv_folds=cv_folds)
    print(f"   Poly3 R²: {results['poly3'].r2_score:.4f}")

    # 4. Log-linear (for exponential decay)
    print("4. Training log-linear regression...")
    results["log_linear"] = predictor.fit_regression(RegressionType.LOG_LINEAR, cv_folds=cv_folds)
    print(f"   Log-linear R²: {results['log_linear'].r2_score:.4f}")

    # 5. Exponential regression
    print("5. Training exponential regression...")
    results["exponential"] = predictor.fit_regression(RegressionType.EXPONENTIAL, cv_folds=cv_folds)
    print(f"   Exponential R²: {results['exponential'].r2_score:.4f}")

    # 6. Kernel regression
    print("6. Training RBF kernel regression...")
    results["kernel_rbf"] = predictor.fit_regression(RegressionType.KERNEL, kernel="rbf", cv_folds=min(5, cv_folds))
    print(f"   Kernel RBF R²: {results['kernel_rbf'].r2_score:.4f}")

    print("7. Training linear kernel regression...")
    results["kernel_linear"] = predictor.fit_regression(RegressionType.KERNEL, kernel="linear", cv_folds=min(5, cv_folds))
    print(f"   Kernel Linear R²: {results['kernel_linear'].r2_score:.4f}")

    # 8. Neural networks (with more data, we can try bigger networks)
    print("8. Training neural network (small)...")
    results["neural_net_small"] = predictor.fit_neural_network(
        hidden_layers=[32, 16], activation="relu", epochs=500, early_stopping=True
    )
    print(f"   Neural net (small) R²: {results['neural_net_small'].r2_score:.4f}")

    if len(all_epochs_remaining) >= 100:
        print("9. Training neural network (large)...")
        results["neural_net_large"] = predictor.fit_neural_network(
            hidden_layers=[64, 32, 16], activation="relu", epochs=1000, early_stopping=True
        )
        print(f"   Neural net (large) R²: {results['neural_net_large'].r2_score:.4f}")
    else:
        print("9. Skipping large neural network (insufficient data)")

    # 10. Ensemble methods
    print("10. Training random forest...")
    results["random_forest"] = predictor.fit_ensemble(n_estimators=100, method="random_forest")
    print(f"    Random forest R²: {results['random_forest'].r2_score:.4f}")

    print("11. Training gradient boosting...")
    results["gradient_boosting"] = predictor.fit_ensemble(n_estimators=100, method="gradient_boosting")
    print(f"    Gradient boosting R²: {results['gradient_boosting'].r2_score:.4f}")

    print("\n=== Model comparison and analysis ===")
    # Compare models
    model_comparison = predictor.compare_models(metric="r2")
    print("Model R² scores:")
    for model, r2 in sorted(model_comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {r2:.4f}")

    print("\nPerforming feature selection...")
    # Feature selection - find most important features
    important_features = predictor.feature_selection(method="lasso", n_features=min(10, features.shape[1]), alpha=0.1)
    print(f"  Selected {len(important_features)} most important features")

    # Also try correlation-based feature selection
    corr_features = predictor.feature_selection(method="correlation", n_features=min(10, features.shape[1]))
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
    print(f"\nComputing bootstrap confidence intervals for {best_model_name}...")
    confidence_intervals = predictor.bootstrap_confidence(model=best_model_name, n_bootstrap=200, confidence=0.95)
    print(f"  Bootstrap completed with {confidence_intervals.get('n_successful_bootstraps', 'unknown')} successful samples")
    
    # Generate human-readable feature names for interpretation
    feature_map = map_features_to_indices(predictor) 
    # Call the new interpretation function for the best model
    predictor.interpret_best_model(best_model_name, feature_map, trace_dir)

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
            "feature_count": features.shape[1],
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
            "cross_val_scores": results[results_key].cross_val_scores.tolist() if results[results_key].cross_val_scores is not None else None
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
        "prediction_confidence": {
            "lower_bound": confidence_intervals["lower"].tolist() if "lower" in confidence_intervals else None,
            "upper_bound": confidence_intervals["upper"].tolist() if "upper" in confidence_intervals else None,
            "bootstrap_samples": confidence_intervals.get('n_successful_bootstraps', 0)
        },
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
    predictor.visualize_results(plot_type="all", save_path=fig_path)
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


def map_features_to_indices(predictor: ConvergencePredictor) -> Dict[int, str]:
    """
    Map feature indices to their mathematical definitions.
    """
    n_params = predictor.data.initial_params.shape[1]
    param_names = [f"p{i}" for i in range(n_params)]
    
    feature_index_map = {}
    current_idx = 0
    
    # Distance metrics (assuming L1, L2, cosine, parameter-wise are used)
    feature_index_map[current_idx] = "L1_distance"
    current_idx += 1
    feature_index_map[current_idx] = "L2_distance" 
    current_idx += 1
    feature_index_map[current_idx] = "cosine_distance"
    current_idx += 1
    
    # Parameter-wise distances
    for i in range(n_params):
        feature_index_map[current_idx] = f"abs_diff_{param_names[i]}"
        current_idx += 1
    
    # Raw differences
    for i in range(n_params):
        feature_index_map[current_idx] = f"diff_{param_names[i]}"
        current_idx += 1
    
    # Ratios
    for i in range(n_params):
        feature_index_map[current_idx] = f"ratio_{param_names[i]}"
        current_idx += 1
    
    # Log transforms
    for i in range(n_params):
        feature_index_map[current_idx] = f"log_diff_{param_names[i]}"
        current_idx += 1
        
    # Interactions
    for i in range(n_params):
        for j in range(i + 1, n_params):
            feature_index_map[current_idx] = f"interaction_{param_names[i]}_{param_names[j]}"
            current_idx += 1
    
    return feature_index_map


def extract_detailed_linear_equation(predictor: ConvergencePredictor, results: Dict) -> str:
    """
    Extract the full linear equation with proper feature names.
    """
    if "linear" not in results:
        return "Linear model not found"
    
    linear_model = predictor.models["linear"]
    weights = linear_model.weight.data.cpu().numpy().flatten()
    bias = linear_model.bias.data.cpu().numpy()[0]
    
    # Get feature names
    feature_map = map_features_to_indices(predictor)
    
    # Build equation
    equation_parts = [f"{bias:.4f}"]
    
    for i, weight in enumerate(weights):
        if abs(weight) > 1e-6:  # Only include significant terms
            feature_name = feature_map.get(i, f"feature_{i}")
            sign = "+" if weight >= 0 else "-"
            equation_parts.append(f" {sign} {abs(weight):.4f} × {feature_name}")
    
    return "epochs_remaining = " + "".join(equation_parts)

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

    print("\n" + "="*80)
    print("MATHEMATICAL EQUATIONS FOR CONVERGENCE PREDICTION")
    print("="*80)

    print("\n=== FEATURE ENGINEERING EQUATIONS ===")
    print("The following synthetic features are computed from raw parameters:")

    print("\nDistance Metrics:")
    print("  L1_distance = |p0_init - p0_final| + |p1_init - p1_final| + |p2_init - p2_final|")
    print("  L2_distance = sqrt((p0_init - p0_final)² + (p1_init - p1_final)² + (p2_init - p2_final)²)")
    print("  cosine_distance = 1 - (initial_params · final_params) / (||initial_params|| × ||final_params||)")

    print("\nRaw Parameter Differences:")
    print("  diff_p0 = p0_init - p0_final")
    print("  diff_p1 = p1_init - p1_final") 
    print("  diff_p2 = p2_init - p2_final")

    print("\nParameter Ratios:")
    print("  ratio_p0 = p0_init / (p0_final + 1e-8)")
    print("  ratio_p1 = p1_init / (p1_final + 1e-8)")
    print("  ratio_p2 = p2_init / (p2_final + 1e-8)")

    print("\nLog Transforms:")
    print("  log_diff_p0 = log(|p0_init - p0_final| + 1e-8)")
    print("  log_diff_p1 = log(|p1_init - p1_final| + 1e-8)")
    print("  log_diff_p2 = log(|p2_init - p2_final| + 1e-8)")

    print("\nInteraction Terms:")
    print("  interaction_p0_p1 = diff_p0 × diff_p1")
    print("  interaction_p0_p2 = diff_p0 × diff_p2")
    print("  interaction_p1_p2 = diff_p1 × diff_p2")

    print("\n=== CONVERGENCE PREDICTION EQUATIONS ===")

    # Print all model equations with their accuracy
    model_comparison = analysis_results['model_comparison']
    sorted_models = sorted(model_comparison.items(), key=lambda x: x[1], reverse=True)

    print("\nAll Trained Models (ranked by accuracy):")
    for i, (model_name, r2_score) in enumerate(sorted_models):
        print(f"\n{i+1}. {model_name.upper()} (R² = {r2_score:.4f}):")
        
        if "linear" in model_name and "log" not in model_name and "kernel" not in model_name:
            print("   epochs_remaining = β₀ + β₁×L1_distance + β₂×L2_distance + β₃×cosine_distance + ...")
            print("   [Linear combination of all 18 features]")
            
        elif "poly" in model_name:
            degree = "2" if "poly_2" in model_name else "3"
            print(f"   epochs_remaining = polynomial expansion of degree {degree}")
            print("   [Includes squared terms, cubic terms, and cross-products of features]")
            
        elif "log_linear" in model_name:
            print("   log(epochs_remaining) = β₀ + β₁×L1_distance + β₂×L2_distance + ...")
            print("   epochs_remaining = exp(β₀ + β₁×L1_distance + β₂×L2_distance + ...)")
            
        elif "exponential" in model_name:
            print("   epochs_remaining = exp(β₀ + β₁×L1_distance + β₂×L2_distance + ...)")
            
        elif "kernel_rbf" in model_name:
            print("   epochs_remaining = Σᵢ αᵢ × exp(-γ||x - xᵢ||²)")
            print("   [Radial Basis Function kernel with Gaussian similarity]")
            
        elif "kernel_linear" in model_name:
            print("   epochs_remaining = Σᵢ αᵢ × (x · xᵢ)")
            print("   [Linear kernel regression]")
            
        elif "neural_net" in model_name or "nn_" in model_name:
            if "small" in model_name or "2layer" in model_name:
                print("   epochs_remaining = NN([32, 16] hidden units)")
            else:
                print("   epochs_remaining = NN([64, 32, 16] hidden units)")
            print("   [Multi-layer neural network with ReLU activations]")
            
        elif "random_forest" in model_name:
            print("   epochs_remaining = average of 100 decision trees")
            print("   [Ensemble of decision trees with feature bagging]")
            
        elif "gradient_boosting" in model_name:
            print("   epochs_remaining = Σₜ learning_rate × tree_t(features)")
            print("   [Sequential ensemble of 100 boosted decision trees]")
            
        else:
            print("   [Complex non-linear model]")

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

