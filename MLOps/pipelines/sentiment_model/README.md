# MLOps Sentiment Model Training Pipeline

This directory contains the scripts and configurations for the sentiment model training pipeline of the FinAI_algo project. The primary goal of this pipeline is to fine-tune Large Language Models (LLMs) like LLaMA-3 (or others specified in FinGPT) using LoRA (Low-Rank Adaptation) for financial sentiment analysis.

## Files

*   **`__init__.py`**: (If this directory becomes a sub-package).
*   **`train_sentiment_model.py`**:
    *   The main script that orchestrates the sentiment model fine-tuning process.
    *   **Functionality:**
        1.  **Loads Configurations:** Reads settings from:
            *   `MLOps/config/common/global_vars.yaml` (for data paths, MLflow settings).
            *   `MLOps/config/sentiment_models/<model_type>_params.yaml` (e.g., `llama3_8b_lora_params.yaml` for model identification, quantization, LoRA config, training arguments, data config).
        2.  **Loads Preprocessed Data:**
            *   Reads preprocessed text data suitable for sentiment model fine-tuning. This data might come from `data/processed/` (e.g., `sentiment_train_dataset.csv`, `sentiment_validation_dataset.csv`) or be prepared by adapting scripts from FinGPT (like `FinNLP/fingpt/FinGPT_Sentiment_Analysis_v3/data/making_data.ipynb`).
            *   The data should typically have columns for text and corresponding sentiment labels.
        3.  **Initializes Tokenizer and Model:**
            *   Loads the tokenizer and base LLM specified in the configuration (e.g., from Hugging Face Hub).
            *   Applies quantization (e.g., 4-bit or 8-bit using `bitsandbytes`) to the base model if configured.
            *   Sets up the PEFT (Parameter-Efficient Fine-Tuning) configuration for LoRA.
        4.  **Experiment Tracking Setup (MLflow):**
            *   Initializes an MLflow run using utilities from `MLOps/experiment_tracking/mlflow_utils.py`.
            *   Logs hyperparameters (LoRA config, training args, quantization config), data configuration, and relevant tags.
        5.  **Fine-Tunes Model:**
            *   Utilizes Hugging Face `Trainer` API (or a custom training loop if necessary) to fine-tune the LLM with LoRA on the prepared sentiment dataset.
            *   The `Trainer` handles the training loop, evaluation on a validation set, and checkpointing.
            *   During training, logs metrics (e.g., training/validation loss, accuracy, F1-score) to MLflow.
        6.  **Saves Trained Model (LoRA Adapters):**
            *   Saves the trained LoRA adapter weights and configuration to a specified path, typically within `MLOps/model_registry/sentiment_models/` or lets MLflow handle model artifact logging.
        7.  **Logs Model to MLflow:**
            *   Uses `mlflow_utils.log_model()` to log the trained LoRA adapters (and potentially a wrapper to load the base model with adapters) to MLflow. This can also register the model in the MLflow Model Registry.
    *   **Execution:**
        ```bash
        python MLOps/pipelines/sentiment_model/train_sentiment_model.py \
          --global_config_path MLOps/config/common/global_vars.yaml \
          --model_params_path MLOps/config/sentiment_models/llama3_8b_lora_params.yaml \
          # --output_model_dir MLOps/model_registry/sentiment_models/llama3_lora_run1 # Optional
        ```

## Key Components & Libraries

*   **LLMs:** LLaMA-3, Llama-2, or other models supported by FinGPT and Hugging Face Transformers.
*   **Fine-tuning Technique:** LoRA (Low-Rank Adaptation) via Hugging Face PEFT library.
*   **Training Framework:** Hugging Face Transformers (`Trainer` API).
*   **Quantization:** `bitsandbytes` for 4-bit/8-bit quantization (QLoRA).
*   **Experiment Tracking:** MLflow, using `MLOps/experiment_tracking/mlflow_utils.py`.
*   **Data:** Preprocessed financial news text with sentiment labels.

## Output

*   **Trained LoRA Adapters:** Saved adapter weights (`adapter_model.bin`) and configuration (`adapter_config.json`).
*   **MLflow Run:** An MLflow run containing:
    *   Logged hyperparameters (LoRA, quantization, training arguments).
    *   Logged training and evaluation metrics.
    *   The trained LoRA model artifacts.
    *   Configuration files as artifacts.
    *   (Optionally) Evaluation reports or plots.

## Dependencies

*   Pandas, NumPy
*   PyYAML
*   Hugging Face Suite: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`
*   PyTorch (or TensorFlow, depending on the LLM and training script adaptation)
*   MLflow
*   DVC (for accessing DVC-tracked training data)

## Future Enhancements

*   Automated data preparation scripts adapted from FinGPT notebooks.
*   Hyperparameter optimization for LoRA and training arguments (e.g., using Optuna, Ray Tune).
*   More comprehensive evaluation metrics and error analysis.
*   Support for different PEFT techniques beyond LoRA.
*   Integration with data labeling tools if custom sentiment datasets are being created/refined.