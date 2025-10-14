# MLOps/pipelines/sentiment_model/train_sentiment_model.py

import logging
import logging.config
import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from functools import partial
import torch # Added for bnb_4bit_compute_dtype
from sklearn.model_selection import train_test_split # Added for splitting data

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Transformers and PEFT imports (similar to FinGPT training scripts)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig # Added for 4-bit quantization
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset, DatasetDict # Updated import

# MLOps utilities (to be created or placeholder)
# from MLOps.experiment_tracking.mlflow_utils import start_mlflow_run, log_params, log_metrics, log_model # Placeholder
# Using the same placeholder as in train_rl_agent.py for now
class MLflowUtilsPlaceholder:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        # logger.info for placeholder init can be noisy if logger is not fully set up when this class is defined.

    def start_mlflow_run(self, experiment_name, run_name):
        logger.info(f"MLP_PH: Starting run '{run_name}' in experiment '{experiment_name}'.")
        class MockRun:
            def __init__(self):
                self.info = type('info', (object,), {'run_id': 'mock_run_id'}) # Mock run_id
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockRun()

    def log_params(self, params_dict):
        logger.info(f"MLP_PH: Logging parameters: {params_dict}")

    def log_metrics(self, metrics_dict, step=None):
        logger.info(f"MLP_PH: Logging metrics: {metrics_dict} at step {step if step else 'N/A'}")

    def log_model(self, model, artifact_path): # model is not used in placeholder
        logger.info(f"MLP_PH: Logging model to artifact path: {artifact_path}")

    def log_artifact(self, local_path, artifact_path=None):
        logger.info(f"MLP_PH: Logging artifact from {local_path} to {artifact_path if artifact_path else ''}")

mlflow_utils = MLflowUtilsPlaceholder()


# Configure logging
# Ensure logging_config.py is correctly located and configured
try:
    logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
    if os.path.exists(logging_config_path):
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
    else:
        # Fallback basic logging if config file is missing
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.warning(f"Logging config file not found at {logging_config_path}. Using basicConfig.")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.error(f"Error configuring logging from file: {e}. Using basicConfig.")

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    logger.debug(f"Loading YAML configuration from: {config_path}")
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise

# Function: load_sentiment_data
# Description: Loads sentiment data for model training. It can load from a single CSV/JSONL file
#              (and perform a train/test split) or from a directory containing pre-split
#              train and test files (train.csv/test.csv or train.jsonl/test.jsonl).
#              The data is expected to have 'instruction', 'input', and 'output' columns.
# Input:
#   data_path: String, path to the data file or directory.
#   test_size: Float, proportion of the dataset to include in the test split if loading a single file (default 0.2).
#   random_state: Integer, random state for reproducible train/test splits (default 42).
# Output: A Hugging Face `datasets.DatasetDict` containing 'train' and optionally 'test' datasets.
# Dependencies: logging, os, pandas (pd), sklearn.model_selection.train_test_split, datasets.Dataset, datasets.DatasetDict.
def load_sentiment_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> DatasetDict:
    logger.debug(f"Loading sentiment data from: {data_path}")
    """
    Loads sentiment data from a CSV or JSONL file, or a directory with train/test splits.
    Assumes data has 'instruction', 'input', and 'output' columns.
    Converts loaded data into a Hugging Face datasets.DatasetDict.
    """
    logger.info(f"Attempting to load sentiment dataset from: {data_path}")
    required_columns = ['instruction', 'input', 'output']

    try:
        if os.path.isdir(data_path):
            train_path_csv = os.path.join(data_path, "train.csv")
            test_path_csv = os.path.join(data_path, "test.csv")
            train_path_jsonl = os.path.join(data_path, "train.jsonl")
            test_path_jsonl = os.path.join(data_path, "test.jsonl")

            if os.path.exists(train_path_csv) and os.path.exists(test_path_csv):
                logger.info(f"Loading pre-split CSV data from directory: {data_path}")
                train_df = pd.read_csv(train_path_csv)
                test_df = pd.read_csv(test_path_csv)
            elif os.path.exists(train_path_jsonl) and os.path.exists(test_path_jsonl):
                logger.info(f"Loading pre-split JSONL data from directory: {data_path}")
                train_df = pd.read_json(train_path_jsonl, lines=True)
                test_df = pd.read_json(test_path_jsonl, lines=True)
            else:
                raise FileNotFoundError(f"No train/test CSV or JSONL files found in directory: {data_path}")

            for col in required_columns:
                if col not in train_df.columns or col not in test_df.columns:
                    raise ValueError(f"Missing required column '{col}' in pre-split data files.")
            
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

        elif os.path.isfile(data_path):
            logger.info(f"Loading single data file: {data_path}")
            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            elif data_path.endswith(".jsonl"):
                df = pd.read_json(data_path, lines=True)
            else:
                raise ValueError(f"Unsupported file type: {data_path}. Please use CSV or JSONL.")

            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in data file: {data_path}")

            if test_size > 0:
                train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
                train_dataset = Dataset.from_pandas(train_df)
                test_dataset = Dataset.from_pandas(test_df)
                dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
                logger.info(f"Data split into train ({len(train_df)}) and test ({len(test_df)}) sets.")
            else:
                dataset = Dataset.from_pandas(df)
                dataset_dict = DatasetDict({'train': dataset}) # No test split if test_size is 0
                logger.info(f"Data loaded as a single 'train' set ({len(df)}). No test split performed.")
        else:
            raise FileNotFoundError(f"Data path not found or is not a file/directory: {data_path}")

        logger.info(f"Sentiment data loaded successfully. Train size: {len(dataset_dict['train'])}, Test size: {len(dataset_dict.get('test', []))}")
        return dataset_dict

    except FileNotFoundError:
        logger.error(f"Sentiment data file or directory not found: {data_path}")
        raise
    except ValueError as e:
        logger.error(f"Error loading or processing sentiment data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading sentiment data: {e}", exc_info=True)
        raise

# Tokenization function - adapt from FinNLP/fingpt/FinGPT_Benchmark/utils.py (tokenize function)
# This needs to be aligned with how FinGPT prepares data for its specific models.
# Function: tokenize_sentiment_data
# Description: Tokenizes a batch of sentiment data examples (instruction, input, output)
#              for training a causal language model. It constructs a prompt from instruction and input,
#              tokenizes the prompt and output, combines them, adds an EOS token, and creates labels
#              where prompt tokens are masked (-100).
# Input:
#   examples: Dictionary-like object (batch from Hugging Face dataset) containing 'instruction', 'input', 'output' lists.
#   tokenizer: Hugging Face `transformers.PreTrainedTokenizer`.
#   max_length: Integer, maximum sequence length for tokenization.
#   instruct_template_str: String, template for formatting instruction and input into a prompt.
# Output: A dictionary containing 'input_ids', 'labels', and 'attention_mask' lists for the batch.
# Dependencies: None (relies on tokenizer passed as argument).
def tokenize_sentiment_data(examples, tokenizer, max_length, instruct_template_str):
    logger.debug(f"Tokenizing batch of {len(examples['input'])} examples.")
    """Tokenizes sentiment data examples for Causal LM fine-tuning."""
    # This function processes a batch of examples (dictionary of lists)
    
    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for i in range(len(examples['input'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i]
        output_text = examples['output'][i]

        prompt = instruct_template_str.format(instruction=instruction, input=input_text)
        
        # Tokenize prompt
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding=False, add_special_tokens=False)
        prompt_ids = prompt_tokenized['input_ids']
        
        # Tokenize output
        output_tokenized = tokenizer(output_text, truncation=True, max_length=max_length, padding=False, add_special_tokens=False)
        output_ids = output_tokenized['input_ids']

        # Combine for input_ids
        combined_input_ids = prompt_ids + output_ids
        
        # Add EOS token if space allows and it's not already the last token
        if combined_input_ids[-1] != tokenizer.eos_token_id and len(combined_input_ids) < max_length:
            combined_input_ids.append(tokenizer.eos_token_id)
        
        # Truncate if exceeds max_length
        combined_input_ids = combined_input_ids[:max_length]
        
        # Create labels: -100 for prompt tokens, actual tokens for output part
        # Ensure prompt_ids are also truncated if the prompt itself is too long
        truncated_prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length, padding=False, add_special_tokens=False)['input_ids'])
        
        labels = ([-100] * truncated_prompt_len) + output_ids
        if combined_input_ids[-1] == tokenizer.eos_token_id and len(labels) < len(combined_input_ids):
             labels.append(tokenizer.eos_token_id) # Label for EOS token
        
        labels = labels[:max_length] # Truncate labels to match input_ids length

        # Ensure labels and input_ids have the same length after all processing
        # This is critical. If input_ids were truncated, labels must also be.
        # If prompt was very long, output_ids part in labels might be empty or truncated.
        if len(labels) > len(combined_input_ids):
            labels = labels[:len(combined_input_ids)]
        elif len(labels) < len(combined_input_ids): # Should not happen if logic is correct
            logger.warning("Labels shorter than input_ids after processing, padding labels with -100. This might indicate an issue.")
            labels.extend([-100] * (len(combined_input_ids) - len(labels)))


        attention_mask = [1] * len(combined_input_ids)

        input_ids_batch.append(combined_input_ids)
        labels_batch.append(labels)
        attention_mask_batch.append(attention_mask)
        
    return {
        'input_ids': input_ids_batch,
        'labels': labels_batch,
        'attention_mask': attention_mask_batch,
    }


def train_sentiment_model(
    dataset: datasets.DatasetDict,
    model_config: dict, # Combined general_config and hyperparameters from sentiment_model_config
    model_output_dir_base: str
):
    logger.debug(f"Initiating sentiment model fine-tuning.")
    """
    Fine-tunes a sentiment analysis model using PEFT (LoRA).
    Adapts logic from FinGPT training scripts.
    """
    logger.info("Starting sentiment model fine-tuning.")

    general_config = model_config.get('general_config', {})
    hyperparameters = model_config.get('hyperparameters', {})

    base_model_name = general_config.get('base_model_name_or_path')
    tokenizer_name = general_config.get('tokenizer_name_or_path', base_model_name)
    
    if not base_model_name:
        logger.error("base_model_name_or_path not specified in sentiment model configuration.")
        raise ValueError("base_model_name_or_path is required.")

    # Resolve local model paths if 'models/' prefix is used
    if base_model_name.startswith("models/"):
        base_model_name = os.path.join(project_root, base_model_name)
    if tokenizer_name.startswith("models/"):
        tokenizer_name = os.path.join(project_root, tokenizer_name)

    logger.info(f"Loading base model: {base_model_name}")
    model_load_kwargs = {}
    quant_config_params = hyperparameters.get('quantization_config', {}) # Check under hyperparameters
    if not quant_config_params: # Fallback to top-level if defined there (older config structure)
        quant_config_params = model_config.get('quantization_config', {})

    torch_dtype_str = hyperparameters.get('torch_dtype', 'auto') # Check under hyperparameters
    if torch_dtype_str == 'auto': # Fallback to top-level
        torch_dtype_str = model_config.get('torch_dtype', 'auto')
    if torch_dtype_str != 'auto':
        try:
            model_load_kwargs['torch_dtype'] = getattr(torch, torch_dtype_str)
            logger.info(f"Setting model torch_dtype to: {torch_dtype_str}")
        except AttributeError:
            logger.warning(f"torch_dtype '{torch_dtype_str}' not recognized. Using transformers default.")

    if quant_config_params.get('load_in_8bit', False): # Default to False if not present
        model_load_kwargs['load_in_8bit'] = True
        model_load_kwargs['device_map'] = general_config.get('device_map', "auto") # device_map from general_config
        logger.info("8-bit quantization enabled.")
    elif quant_config_params.get('load_in_4bit', False): # Default to False
        logger.info("4-bit quantization configured.")
        bnb_compute_dtype_str = quant_config_params.get('bnb_4bit_compute_dtype', "float16") # from quant_config
        try:
            bnb_compute_dtype = getattr(torch, bnb_compute_dtype_str)
        except AttributeError:
            logger.warning(f"bnb_4bit_compute_dtype '{bnb_compute_dtype_str}' not recognized. Defaulting to torch.float16.")
            bnb_compute_dtype = torch.float16 # Fallback

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config_params.get('bnb_4bit_quant_type', "nf4"), # from quant_config
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=quant_config_params.get('bnb_4bit_use_double_quant', False), # from quant_config
        )
        model_load_kwargs['quantization_config'] = bnb_config
        model_load_kwargs['device_map'] = general_config.get('device_map', "auto") # device_map from general_config


    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **model_load_kwargs
    )
    # Device placement if not handled by device_map (e.g. no quantization)
    if not model_load_kwargs.get('device_map'):
        device = hyperparameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Model explicitly moved to device: {device}")


    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer pad_token set to eos_token ('{tokenizer.eos_token}'). Ensure padding_side is 'left' for generation if this model requires it.")


    # Tokenize dataset
    # instruction_template from general_config or a default
    instruct_template = general_config.get('instruction_template_example', "Instruction: {instruction}\nInput: {input}\nAnswer: ")
    default_max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 512
    max_seq_length = hyperparameters.get('max_seq_length', default_max_len)
    
    logger.info(f"Tokenizing dataset with max_seq_length: {max_seq_length} and instruction template: '{instruct_template[:100]}...'") # Log truncated template
    tokenized_dataset = dataset.map(
        partial(tokenize_sentiment_data, tokenizer=tokenizer, max_length=max_seq_length, instruct_template_str=instruct_template),
        batched=True, # Process examples in batches
        num_proc=hyperparameters.get('preprocessing_num_workers', os.cpu_count() // 2 or 1),
        remove_columns=dataset["train"].column_names # Remove original columns after tokenization
    )
    logger.info("Dataset tokenized.")

    # PEFT (LoRA) configuration
    lora_config_params = hyperparameters.get('lora_config', {}) # Check under hyperparameters
    if not lora_config_params: # Fallback to top-level
        lora_config_params = model_config.get('lora_config', {})

    # Determine target_modules based on base_model_name (e.g. from FinNLP/fingpt/FinGPT_Benchmark/utils.py)
    # This requires a mapping similar to lora_module_dict in FinGPT's utils.
    # For now, using a placeholder or a common default.
    # A more robust solution would be to have this mapping available.
    # Example for Llama: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Example for ChatGLM2: ["query_key_value"]
    # This should be part of the sentiment_model_config.yaml ideally.
    target_modules_default = ["q_proj", "v_proj"] # A common minimal set, adjust per model
    if "chatglm2" in base_model_name.lower():
        target_modules_default = ["query_key_value"]
    elif "llama" in base_model_name.lower():
         target_modules_default = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


    peft_config = LoraConfig(
        task_type=TaskType[lora_config_params.get('task_type', 'CAUSAL_LM')],
        r=lora_config_params.get('r', 8), # Default from config
        lora_alpha=lora_config_params.get('lora_alpha', 32), # Default from config
        lora_dropout=lora_config_params.get('lora_dropout', 0.1), # Default from config
        target_modules=lora_config_params.get('target_modules', target_modules_default),
        bias=lora_config_params.get('bias', 'none'),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("PEFT model configured with LoRA.")

    # TrainingArguments
    # Use a sub-directory for each run to store checkpoints, logs, etc.
    run_specific_output_dir = os.path.join(model_output_dir_base, f"{base_model_name.replace('/', '_')}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    trainer_args_dict = {
        "output_dir": run_specific_output_dir,
        "logging_dir": os.path.join(run_specific_output_dir, 'tf_logs'), # Tensorboard logs
        "logging_strategy": hyperparameters.get('logging_strategy', "steps"),
        "logging_steps": hyperparameters.get('logging_steps', 100),
        "num_train_epochs": hyperparameters.get('num_train_epochs', 2.0), # Can be float
        "per_device_train_batch_size": hyperparameters.get('per_device_train_batch_size', 4),
        "per_device_eval_batch_size": hyperparameters.get('per_device_eval_batch_size', 4),
        "gradient_accumulation_steps": hyperparameters.get('gradient_accumulation_steps', 8),
        "learning_rate": hyperparameters.get('learning_rate', 1e-4), # from config
        "weight_decay": hyperparameters.get('weight_decay', 0.01),
        "lr_scheduler_type": hyperparameters.get('lr_scheduler_type', 'cosine'),
        "save_strategy": hyperparameters.get('save_strategy', "steps"), # Save based on steps
        "save_steps": hyperparameters.get('save_steps', 500),
        "save_total_limit": hyperparameters.get('save_total_limit', 2), # Keep last 2 checkpoints
        "evaluation_strategy": hyperparameters.get('evaluation_strategy', "steps" if "test" in tokenized_dataset else "no"),
        "eval_steps": hyperparameters.get('eval_steps', hyperparameters.get('save_steps', 500)), # Eval at save steps
        "fp16": hyperparameters.get('fp16', False), # from config
        "bf16": hyperparameters.get('bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "load_best_model_at_end": hyperparameters.get('load_best_model_at_end', True if "test" in tokenized_dataset and hyperparameters.get('evaluation_strategy') != "no" else False),
        "remove_unused_columns": hyperparameters.get('remove_unused_columns', False), # FinGPT sets to False
        "report_to": "wandb" if os.getenv("WANDB_API_KEY") and mlflow_utils.__class__.__name__ != "MLflowUtilsPlaceholder" else "none", # Enable wandb if API key is set
        "optim": hyperparameters.get("optim", "paged_adamw_8bit" if quant_config_params.get('load_in_8bit') else "adamw_torch"),
        "deepspeed": hyperparameters.get("deepspeed_config_path") if hyperparameters.get("deepspeed_config_path") else None,
    }
    if trainer_args_dict['bf16']: trainer_args_dict['fp16'] = False # bf16 and fp16 are mutually exclusive

    # Warmup steps or ratio
    if 'warmup_ratio' in hyperparameters and hyperparameters['warmup_ratio'] > 0:
        trainer_args_dict['warmup_ratio'] = hyperparameters['warmup_ratio']
        logger.info(f"Using warmup_ratio: {hyperparameters['warmup_ratio']}")
    elif 'warmup_steps' in hyperparameters: # Check if warmup_steps is explicitly set
        trainer_args_dict['warmup_steps'] = hyperparameters['warmup_steps']
        logger.info(f"Using warmup_steps: {hyperparameters['warmup_steps']}")
    else: # Default if neither is set
        trainer_args_dict['warmup_ratio'] = 0.03
        logger.info(f"Using default warmup_ratio: 0.03")
    
    trainer_args = TrainingArguments(**trainer_args_dict)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("test"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting model training...")
    train_result = trainer.train()
    logger.info(f"Training completed. Train metrics: {train_result.metrics}")
    
    # Instead of trainer.save_model(), for PEFT models, it's common to save the adapter
    # The trainer will save checkpoints which include the adapter.
    # To save the final adapter explicitly:
    # model.save_pretrained(os.path.join(run_specific_output_dir, "final_adapter"))
    # tokenizer.save_pretrained(os.path.join(run_specific_output_dir, "final_adapter_tokenizer"))
    # For now, rely on trainer's checkpointing and load_best_model_at_end.
    # The best model (adapter) will be in the best checkpoint directory.

    mlflow_utils.log_params(trainer_args_dict) # Log the actual args used
    mlflow_utils.log_metrics(train_result.metrics, step=trainer.state.global_step if hasattr(trainer.state, 'global_step') else None)
    
    # Determine the path of the best model if load_best_model_at_end is True
    final_adapter_path = run_specific_output_dir
    if trainer_args.load_best_model_at_end and trainer.state.best_model_checkpoint:
        final_adapter_path = trainer.state.best_model_checkpoint
        logger.info(f"Best model checkpoint (LoRA adapter) is at: {final_adapter_path}")
        # Explicitly save the best adapter to a consistent name if desired, or log this path.
        # model.save_pretrained(os.path.join(run_specific_output_dir, "best_lora_adapter"))
    else:
        # If not loading best, the last saved checkpoint is effectively the final model.
        # Or explicitly save the final state of the adapter.
        model.save_pretrained(os.path.join(run_specific_output_dir, "final_lora_adapter"))
        final_adapter_path = os.path.join(run_specific_output_dir, "final_lora_adapter")

    mlflow_utils.log_artifact(final_adapter_path, "trained_lora_adapter_files")
    
    return final_adapter_path, trainer


# Function: main
# Description: Main orchestrator for the sentiment model training pipeline.
#              It loads configurations, sentiment data, trains the sentiment model using LoRA,
#              evaluates it (if a test set is available), and logs results/artifacts using (placeholder) MLflow.
# Input: None (reads configuration and data paths from predefined locations and environment variables).
# Output: None. Executes the sentiment model training pipeline.
# Dependencies: logging, os, sys, datetime, load_config (local function),
#               load_sentiment_data (local function), train_sentiment_model (local function),
#               mlflow_utils (global placeholder).
# Globals: project_root.
def main():
    logger.debug(f"Main orchestrator for the sentiment model training pipeline started.")
    logger.info("MLOps Sentiment Model Training Pipeline started.")

    global_vars_config_path = os.path.join(project_root, "MLOps", "config", "common", "global_vars.yaml")
    # Use environment variable or a default config file name for sentiment model params
    default_sentiment_config_filename = "llama3_8b_lora_params.yaml" # Example default
    sentiment_config_filename_env = os.getenv("SENTIMENT_MODEL_CONFIG_FILE", default_sentiment_config_filename)
    sentiment_model_config_path = os.path.join(project_root, "MLOps", "config", "sentiment_models", sentiment_config_filename_env)
    
    global_vars_config = load_config(global_vars_config_path)
    
    try:
        sentiment_model_config = load_config(sentiment_model_config_path)
    except FileNotFoundError:
        logger.error(f"Sentiment model configuration file not found: {sentiment_model_config_path}. Please ensure it exists or set SENTIMENT_MODEL_CONFIG_FILE env var.")
        sys.exit(1)

    # Get data path from global_vars.yaml, under paths: sentiment_dataset_path: "data/processed/sentiment_data/your_dataset.jsonl"
    paths_config = global_vars_config.get("paths", {})
    sentiment_data_path_relative = paths_config.get("sentiment_dataset_path") # New key in global_vars.yaml
    if not sentiment_data_path_relative:
        logger.error("sentiment_training_data_path not found in global_vars.yaml. Please specify the path to your training data.")
        sys.exit(1)
    
    sentiment_data_path = os.path.join(project_root, sentiment_data_path_relative)
    if not os.path.exists(sentiment_data_path): # Check if resolved path exists
        logger.error(f"Specified sentiment_training_data_path does not exist: {sentiment_data_path}")
        sys.exit(1)

    model_output_dir_base_rel = paths_config.get("model_registry_dir", "MLOps/model_registry") # Store under model_registry
    model_output_dir_base_abs = os.path.join(project_root, model_output_dir_base_rel, "sentiment_models")
    os.makedirs(model_output_dir_base_abs, exist_ok=True)

    # MLflow experiment setup
    experiment_name = sentiment_model_config.get("general_config", {}).get("mlflow_experiment_name", "FinAI_Sentiment_FineTuning")
    # Use base model name from its config for run name
    base_model_name_from_cfg = sentiment_model_config.get('general_config', {}).get('base_model_name_or_path', 'sentiment_model')
    base_model_short_name = base_model_name_from_cfg.split('/')[-1] # Get last part of path
    run_name = f"FineTune_{base_model_short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow_utils.start_mlflow_run(experiment_name, run_name) as run: # run object is MockRun here
        mlflow_utils.log_params({"sentiment_model_config_file": sentiment_config_filename_env})
        mlflow_utils.log_params({"full_sentiment_model_config": sentiment_model_config}) # Log the whole config
        mlflow_utils.log_params({"sentiment_data_path_used": sentiment_data_path_relative})
        
        trainer_instance = None
        try:
            # Training parameters are now directly under 'hyperparameters' in the config
            current_hyperparameters = sentiment_model_config.get('hyperparameters', {})
            
            dataset = load_sentiment_data(
                sentiment_data_path,
                test_size=current_hyperparameters.get('test_split_ratio', 0.1)
            )
            
            if not dataset or not dataset.get('train'):
                logger.error("Sentiment training data is empty or not loaded correctly.")
                raise ValueError("Sentiment training data is insufficient.")
            if current_hyperparameters.get('evaluation_strategy') != "no" and not dataset.get('test'):
                 logger.warning("Evaluation strategy requires a test set, but no test set was loaded/created. Evaluation will be skipped by Trainer.")


            # Pass the whole sentiment_model_config which contains general_config and hyperparameters
            final_adapter_path, trainer_instance = train_sentiment_model(
                dataset,
                sentiment_model_config, # Pass the whole loaded config
                model_output_dir_base_abs
            )
            
            # Log the relative path to the adapter for portability
            final_adapter_path_relative = os.path.relpath(final_adapter_path, project_root)
            mlflow_utils.log_params({"final_adapter_path": final_adapter_path_relative.replace("\\", "/")})

            if trainer_instance and dataset.get("test") and trainer_instance.args.evaluation_strategy != "no":
                logger.info("Evaluating model on the test set (using best model if load_best_model_at_end=True)...")
                eval_results = trainer_instance.evaluate(eval_dataset=dataset["test"]) # Explicitly pass test dataset
                logger.info(f"Evaluation results on test set: {eval_results}")
                mlflow_utils.log_metrics(eval_results, step=trainer_instance.state.global_step if hasattr(trainer_instance.state, 'global_step') else None)
            else:
                logger.info("Skipping final evaluation on test set as no test data is available or evaluation_strategy is 'no'.")

            logger.info(f"MLOps Sentiment Model Training Pipeline finished successfully. Final LoRA adapter saved in: {final_adapter_path}")

        except Exception as e:
            logger.error(f"MLOps Sentiment Model Training Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()