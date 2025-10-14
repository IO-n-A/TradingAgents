# MLOps/pipelines/data_ingestion/__init__.py

"""
This sub-package handles the data ingestion pipeline for the FinAI_algo project.

Modules:
- data_ingestion_pipeline: Contains the main script and functions for fetching
                           raw financial data and news data.
"""

from .data_ingestion_pipeline import run_data_ingestion_pipeline

import logging
logger = logging.getLogger(__name__)
logger.info("MLOps.pipelines.data_ingestion package initialized.")

__all__ = ['run_data_ingestion_pipeline']