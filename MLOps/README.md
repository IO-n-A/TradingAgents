# TradingAgents MLOps

The `MLOps` package orchestrates data, model, and deployment workflows for the TradingAgents project. It brings together configuration management, automated pipelines, experiment tracking, model registry practices, and monitoring assets required to move trading strategies from research to operations.

## Responsibilities
- Automate ingestion, feature engineering, and training routines for sentiment models and RL agents.
- Centralize configuration, hyperparameters, and environment definitions used by FinRL-based components.
- Provide orchestration hooks, deployment entry points, and monitoring/alerting assets for paper trading and backtesting.
- Maintain experiment tracking helpers and conventions for registering models with MLflow or filesystem-based registries.
- Supply tests and notebooks to validate pipelines and perform exploratory analysis.

## Repository Layout
| Path | Description |
| --- | --- |
| `config/` | Project-wide YAML configs for global paths, data sources, trading environments, RL agents, orchestration, and sentiment models. |
| `hyperparameters/` | Default tuning values for RL agents and sentiment models that complement the config files. |
| `pipelines/` | Modular pipelines covering ingestion, feature engineering, model training/fine-tuning, RL agent training, and orchestration helpers (each subdir has its own README). |
| `orchestration/` | High-level orchestrator scripts (e.g., daily data collection, full strategy runs) that stitch together individual pipelines. |
| `deployment/` | Backtesting, paper trading, and future live-trading entry points with instructions for promoting registered models. |
| `monitoring/` | Alerting rules, dashboard definitions (Grafana/Streamlit/Plotly), and scripts for checks such as data drift. |
| `experiment_tracking/` | Utilities for working with MLflow tracking servers and artifact stores. |
| `model_registry/` | Guidance and utilities for managing model versions via MLflow or a filesystem-based fallback. |
| `notebooks/` | Exploratory analyses and documentation supporting pipeline development. |
| `tests/` | Unit and integration tests for pipelines, plus testing guidelines. |
| `README.md` | This document. |

## Core Workflows
- **Data ingestion** (`pipelines/data_ingestion/data_ingestion_pipeline.py`): downloads OHLCV and news data via FinRL utilities, logs metadata to MLflow, and stores raw datasets (DVC hooks available).
- **Feature engineering** (`pipelines/feature_engineering/`): merges financial and sentiment data, adds technical indicators, and produces processed datasets for modeling.
- **Sentiment model fine-tuning** (`pipelines/finetuning/` and `pipelines/sentiment_model/`): prepares datasets and trains LoRA-style LLM sentiment models according to configs in `config/sentiment_models/` and `hyperparameters/`.
- **RL agent training** (`pipelines/rl_agent/`): trains FinRL reinforcement learning agents using environment configs in `config/environments/` and agent parameter files in `config/rl_agents/`.
- **End-to-end orchestration** (`orchestration/*.py`): sequences pipelines for daily data refreshes, sentiment fine-tuning schedules, or full strategy runs.
- **Deployment paths** (`deployment/`): run backtests, deploy to paper trading (e.g., Alpaca), and stage assets for live trading once ready.

## Configuration & Secrets
- Global paths, date ranges, and storage locations live in `config/common/global_vars.yaml`.
- Data source credentials and runtime switches are defined in `config/data_sources.yaml` and (externally) `config/api_keys.yaml` at the repository root.
- Environment definitions for FinRL simulations reside in `config/environments/`.
- RL and sentiment model parameter sets are versioned under `config/rl_agents/`, `config/sentiment_models/`, and `hyperparameters/`.
- Orchestration settings (e.g., which pipelines compose a strategy) are stored in `config/orchestration/`.

## Experiment Tracking & Model Registry
- `experiment_tracking/mlflow_utils.py` abstracts common MLflow operations such as starting runs, logging artifacts, and registering models.
- Training pipelines log metrics, parameters, and artifacts to MLflow, enabling comparison across runs.
- `model_registry/` documents the preferred MLflow Model Registry workflow and offers a filesystem-based alternative for local development.
- Deployment scripts expect registered model names or explicit artifact paths; promote models to the appropriate stage before running paper trading or backtests.

## Monitoring & Observability
- Alertmanager rules in `monitoring/alerts/` define threshold-based notifications for data quality and pipeline health.
- `monitoring/dashboards/` hosts Grafana JSON dashboards, Plotly/PlotVis scripts, and a Streamlit app for interactive monitoring.
- `monitoring/scripts/` contains utilities such as `check_data_drift.py` that can be scheduled alongside orchestrators.

## Related Documentation
- Each major subdirectory ships with its own README for deep dives.
- The repository root README covers the broader TradingAgents architecture and how `MLOps` integrates with the rest of the project.
