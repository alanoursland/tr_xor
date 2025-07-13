# cluster.py - Experiments for clustered XOR
import torch
import models

from configs import get_experiment_config, experiment, ExperimentConfig, TrainingConfig, DataConfig, AnalysisConfig, ExecutionConfig, LoggingConfig
from data import xor_data_centered, xor_labels_T1, accuracy_binary_threshold

# @experiment("abs1_cluster")
# def config_abs1_cluster() -> ExperimentConfig:
#     model = models.Model_Abs1().init_kaiming()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))

#     return ExperimentConfig(
#         model=model,
#         training=TrainingConfig(
#             optimizer=optimizer, loss_function=loss_function, epochs=1000, stop_training_loss_threshold=1e-7
#         ),
#     )