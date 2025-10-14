# MLOps/pipelines/__init__.py

"""
This package contains modules related to MLOps pipelines for the FinAI_algo project.

Sub-packages and modules include:
- data_ingestion: Pipelines for fetching raw data.
- feature_engineering: Pipelines for processing raw data and creating features.
- sentiment_model: Pipelines for training and evaluating sentiment analysis models.
- rl_agent: Pipelines for training and evaluating reinforcement learning agents.
- orchestration: Scripts or configurations for orchestrating the execution of multiple pipelines.

This __init__.py file makes the 'pipelines' directory a Python package.
It can also be used to expose a simplified API from the submodules if needed,
or for package-level initializations.
"""

# Example: You could import key functions or classes here for easier access,
# though it's often kept minimal.
# from .data_ingestion.data_ingestion_pipeline import run_data_ingestion_pipeline
# from .feature_engineering.build_features import run_feature_engineering_pipeline

import logging

logger = logging.getLogger(__name__)
logger.info("MLOps.pipelines package initialized.")

# You can define package-level constants or configurations here if necessary.
# PIPELINE_VERSION = "0.1.0"