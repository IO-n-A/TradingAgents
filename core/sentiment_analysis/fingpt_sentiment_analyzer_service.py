# core/sentiment_analysis/fingpt_sentiment_analyzer_service.py
"""
Generates sentiment scores for processed news text using a fine-tuned FinGPT-style model.
This service loads a pre-trained Hugging Face transformer model for sentiment analysis.
"""
import logging
from typing import List, Dict, Union, Optional
import os
import re # For parsing generated text

# Attempt to import PyTorch, Transformers, PEFT, and Accelerate
try:
    import torch
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModel,  # Changed from AutoModelForSequenceClassification
        BitsAndBytesConfig
    )
    from peft import PeftModel
    import accelerate # Check if accelerate is installed
    TRANSFORMERS_AND_PEFT_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AND_PEFT_AVAILABLE = False
    # Define dummy classes if transformers or peft is not available
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            raise ImportError(f"Transformers/PEFT library not found. Please install it. Error: {e}")
    class AutoModel: # Changed from AutoModelForSequenceClassification
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            raise ImportError(f"Transformers/PEFT library not found. Please install it. Error: {e}")
    class PeftModel:
        @staticmethod
        def from_pretrained(model, model_id, **kwargs):
            raise ImportError(f"PEFT library not found. Please install it. Error: {e}")
    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            raise ImportError(f"bitsandbytes library not found. Please install it. Error: {e}")
    class pipeline:
        def __init__(self, task: str, model, tokenizer, device: int = -1, **kwargs):
            raise ImportError(f"Transformers/PEFT library not found. Please install it. Error: {e}")
        def __call__(self, texts: List[str], **kwargs):
            raise ImportError(f"Transformers/PEFT library not found. Please install it. Error: {e}")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Default instruction template for generative sentiment analysis
DEFAULT_INSTRUCTION_TEMPLATE = (
    "Instruction: Given the financial news, "
    "classify the sentiment of the news into one of the following categories: positive, negative, neutral.\n"
    "Input: {text}\n"
    "Answer: "
)

DEFAULT_MODEL_NAME = "THUDM/chatglm2-6b" # Default base model for Strategy 1

# This class provides sentiment analysis using a Hugging Face transformer model,
# potentially enhanced with LoRA adapters and quantization.
# It can use generative models (like ChatGLM2) for sentiment by parsing their output,
# or standard sequence classification models.
class FinGPTSentimentAnalyzerService:
    """
    A service for generating sentiment scores using a Hugging Face transformer model,
    with support for LoRA adapters and quantization, primarily targeting generative models.

    This class loads a base model and tokenizer, optionally applies LoRA adapters,
    and sets up a pipeline for sentiment analysis. For generative models, it uses a
    text-generation pipeline and parses the output.
    """

    def __init__(
        self,
        base_model_name_or_path: str, # e.g., "THUDM/chatglm2-6b" or "ProsusAI/finbert"
        lora_adapter_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        model_config_path: Optional[str] = None, # For future use with YAML configs
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype_str: str = "torch.float16",
        # If model is generative, this template is used for prompting
        instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
        # Specify model type if it cannot be easily inferred or to override
        model_type: str = "generative" # "generative" or "classification"
    ):
        """
        Initializes the FinGPTSentimentAnalyzerService.

        Args:
            base_model_name_or_path (str): Name/path of the base Hugging Face model.
            lora_adapter_path (Optional[str]): Path to the trained LoRA adapter directory.
            tokenizer_name_or_path (Optional[str]): Name/path of the tokenizer. If None, uses `base_model_name_or_path`.
            model_config_path (Optional[str]): Path to a YAML configuration file (future use).
            device (Optional[str]): Device to run on ('cpu', 'cuda', etc.). Auto-detects if None.
            load_in_4bit (bool): Whether to load the model in 4-bit precision.
            bnb_4bit_quant_type (str): Quantization type for 4-bit (e.g., 'nf4', 'fp4').
            bnb_4bit_use_double_quant (bool): Whether to use double quantization for 4-bit.
            bnb_4bit_compute_dtype_str (str): Compute dtype for 4-bit (e.g., 'torch.float16').
            instruction_template (str): Template for prompting generative models.
            model_type (str): Type of model, "generative" or "classification". Affects pipeline task.
        """
        if not TRANSFORMERS_AND_PEFT_AVAILABLE:
            logger.error("Transformers, PEFT, or Accelerate library not found. Sentiment analysis will not function.")
            print("FinGPTSentimentAnalyzerService cannot be initialized: Core libraries missing.")
            raise ImportError("Transformers, PEFT, and/or Accelerate is not installed. Please install them.")

        self.base_model_name_or_path = base_model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self.tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path else base_model_name_or_path
        self.model_config_path = model_config_path
        self.instruction_template = instruction_template
        self.model_type = model_type.lower()
        self.sentiment_pipeline = None
        self.model = None # Store the loaded model
        self.tokenizer = None # Store the loaded tokenizer

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Target device: {self.device}")

        model_kwargs = {"trust_remote_code": True} # Common for models like ChatGLM
        q_config = None
        if load_in_4bit:
            if not torch.cuda.is_available():
                logger.warning("4-bit quantization (load_in_4bit=True) requires CUDA, but CUDA is not available. Disabling quantization.")
                load_in_4bit = False # Override if no CUDA
            else:
                compute_dtype = getattr(torch, bnb_4bit_compute_dtype_str.split('.')[-1], torch.float16)
                q_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype
                )
                model_kwargs["quantization_config"] = q_config
                model_kwargs["device_map"] = "auto" # device_map needs accelerate
                logger.info(f"4-bit quantization enabled with config: {q_config}")


        try:
            logger.info(f"Loading tokenizer from: {self.tokenizer_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Tokenizer pad_token_id set to eos_token_id: {self.tokenizer.eos_token_id}")

            logger.info(f"Loading base model from: {self.base_model_name_or_path}")
            # Use AutoModel for generative, AutoModelForSequenceClassification for classifiers
            if self.model_type == "generative":
                self.model = AutoModel.from_pretrained(self.base_model_name_or_path, **model_kwargs)
            elif self.model_type == "classification":
                # For classification, device_map might not be standard with AutoModelForSequenceClassification
                # if not using PEFT's prepare_model_for_kbit_training which handles it.
                # If quantizing a classifier, ensure it's handled correctly.
                # For simplicity, if model_type is classification, we assume no advanced quantization for now or it's handled.
                if "device_map" in model_kwargs and not self.lora_adapter_path: # device_map is tricky without PEFT on classifiers
                    logger.warning("device_map='auto' with 'classification' model_type without LoRA might be unstable. Ensure model is on device.")
                # Classification models usually don't need trust_remote_code=True unless custom
                klass_model_kwargs = {k:v for k,v in model_kwargs.items() if k != "trust_remote_code" or "bert" not in self.base_model_name_or_path.lower()}
                if q_config: # If quantization is on for classifier
                     self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name_or_path, **klass_model_kwargs)
                else: # No quantization, ensure it's on device
                     self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name_or_path, **{k:v for k,v in klass_model_kwargs.items() if k != "quantization_config" and k != "device_map"})
                     if "device_map" not in klass_model_kwargs: # if no device_map, move to device manually
                        self.model = self.model.to(self.device)

            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'generative' or 'classification'.")
            logger.info(f"Base model {self.base_model_name_or_path} (type: {self.model_type}) loaded.")

            if self.lora_adapter_path:
                logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                # If base model was not loaded with device_map, ensure it's on the target device before applying adapters
                if "device_map" not in model_kwargs:
                    self.model = self.model.to(self.device)
                
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path, device_map=model_kwargs.get("device_map"))
                logger.info(f"LoRA adapter {self.lora_adapter_path} loaded and applied.")
            elif "device_map" not in model_kwargs and self.model_type == "generative": # Ensure generative model is on device if no device_map
                 self.model = self.model.to(self.device)


            # Determine device for pipeline
            # If device_map is used, model is already on device(s). Pipeline's device param can be -1 or omitted.
            pipeline_device_idx = -1 # Default for CPU or if device_map handled placement
            if not model_kwargs.get("device_map"): # If no device_map was used for model loading
                if self.device.type == 'cuda':
                    pipeline_device_idx = self.device.index if self.device.index is not None else 0
            
            pipeline_task = "text-generation" if self.model_type == "generative" else "sentiment-analysis"
            self.sentiment_pipeline = pipeline(
                task=pipeline_task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device_idx
            )
            logger.info(f"{pipeline_task} pipeline loaded successfully for model: {self.base_model_name_or_path} "
                        f"{'with LoRA: ' + self.lora_adapter_path if self.lora_adapter_path else ''}")

        except Exception as e:
            logger.error(f"Failed to load model, tokenizer, or pipeline: {e}", exc_info=True)
            print(f"Error initializing FinGPTSentimentAnalyzerService: {e}")
            raise RuntimeError(f"Could not initialize sentiment analysis pipeline: {e}")
        
        print(f"FinGPTSentimentAnalyzerService initialized. Model: {self.base_model_name_or_path}, "
              f"LoRA: {self.lora_adapter_path or 'N/A'}, Type: {self.model_type}, Device: {self.model.device if hasattr(self.model, 'device') else self.device}.")


    # This method predicts sentiment for a list of texts.
    # For generative models, it prompts the model and parses the generated text.
    # For classification models, it uses the standard sentiment-analysis pipeline output.
    def predict_sentiment(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        """
        Predicts sentiment for a list of texts.

        Args:
            texts (List[str]): A list of processed (cleaned) text strings.
            batch_size (int): The batch size to use for inference.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries,
            where each dictionary contains 'label' (e.g., 'positive', 'negative', 'neutral')
            and 'score' (the confidence score for that label).
            Returns an empty list if the input is empty or the pipeline is not initialized.
        """
        """
        Predicts sentiment for a list of texts.
        Handles both generative and classification model types.

        Args:
            texts (List[str]): A list of processed (cleaned) text strings.
            batch_size (int): The batch size to use for inference.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries,
            where each dictionary contains 'label' (e.g., 'positive', 'negative', 'neutral')
            and 'score'. Score is 1.0 for generative models if label is found, 0.0 otherwise.
            Returns an empty list if input is empty or pipeline not initialized.
        """
        if not self.sentiment_pipeline:
            logger.error("Sentiment analysis pipeline is not initialized. Cannot predict.")
            print("Sentiment pipeline not initialized. Prediction failed.")
            return []
        if not texts:
            logger.info("Input text list is empty. Returning empty list.")
            return []

        processed_results = []
        try:
            if self.model_type == "generative":
                logger.info(f"Predicting sentiment for {len(texts)} texts using generative model with batch size {batch_size}...")
                prompts = [self.instruction_template.format(text=t) for t in texts]
                
                # text-generation pipeline might not support `batch_size` argument in the same way.
                # It usually processes one by one or handles batching internally if given a list.
                # We pass it, but its effectiveness depends on the specific pipeline implementation.
                # `max_length` for generative pipeline refers to total length (prompt + generation).
                # `max_new_tokens` is better to control generated part.
                pipeline_output = self.sentiment_pipeline(
                    prompts,
                    batch_size=batch_size, # May or may not be used effectively by all text-gen pipelines
                    truncation=True, # Truncate prompt if too long
                    max_new_tokens=15,  # Max tokens for "positive", "negative", "neutral" + buffer
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False # For more deterministic output
                )

                for i, item_preds in enumerate(pipeline_output):
                    # Output format of text-generation pipeline is typically List[Dict[str, str]] per item in input list
                    # e.g., [{'generated_text': 'Instruction: ... Input: ... Answer: positive'}]
                    generated_text_full = item_preds[0]['generated_text'] if isinstance(item_preds, list) and item_preds else \
                                          item_preds['generated_text'] if isinstance(item_preds, dict) else ""

                    answer_prefix_match = re.search(r"Answer:\s*", generated_text_full, re.IGNORECASE)
                    label = "unknown"
                    if answer_prefix_match:
                        answer_part = generated_text_full[answer_prefix_match.end():].strip().lower()
                        # Extract first word, remove punctuation
                        potential_label = re.split(r'[,\.\s]', answer_part)[0]
                        if potential_label in ["positive", "negative", "neutral"]:
                            label = potential_label
                        else:
                            logger.warning(f"Unrecognized label '{potential_label}' from text: '{texts[i]}'. Full generated: '{generated_text_full}'")
                    else:
                        logger.warning(f"Could not parse 'Answer:' prefix for text: '{texts[i]}'. Full generated: '{generated_text_full}'")
                    
                    score = 1.0 if label != "unknown" else 0.0
                    processed_results.append({'label': label, 'score': score, 'raw_generated': generated_text_full})

            elif self.model_type == "classification":
                logger.info(f"Predicting sentiment for {len(texts)} texts using classification model with batch size {batch_size}...")
                # Standard sentiment-analysis pipeline
                results = self.sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=512)
                for res in results:
                    label = res.get('label', 'unknown').lower()
                    # Standardize common label formats (e.g., 'LABEL_0', 'LABEL_1', 'LABEL_2')
                    if label == "label_0" or label == "0": label = "negative"
                    elif label == "label_1" or label == "1": label = "neutral"
                    elif label == "label_2" or label == "2": label = "positive"
                    # FinBERT specific labels are already fine
                    elif label not in ["positive", "negative", "neutral"]:
                        logger.warning(f"Unrecognized label '{label}' from classification model. Raw: {res}")
                        label = "unknown"
                        
                    processed_results.append({
                        'label': label,
                        'score': res.get('score', 0.0)
                    })
            logger.info(f"Sentiment prediction completed for {len(texts)} texts.")

        except Exception as e:
            logger.error(f"Error during sentiment prediction: {e}", exc_info=True)
            print(f"Error during sentiment prediction: {e}")
            # Return partial results if any, or empty
            return processed_results or [] # Ensure it returns a list

        # Sentiment scores have been generated for the input texts.
        # The results provide a label and confidence score (or 1.0/0.0 for generative) for each text.
        print(f"Sentiment prediction successful for {len(texts)} texts. Processed {len(processed_results)} results.")
        return processed_results

if __name__ == '__main__':
    if not TRANSFORMERS_AND_PEFT_AVAILABLE:
        logger.error("Cannot run example: Transformers, PEFT, or Accelerate library not found.")
        print("Please install PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes to run this example fully:")
        print("pip install torch transformers peft accelerate bitsandbytes sentencepiece")
    else:
        # Example Usage:
        sample_texts = [
            "The company reported record profits this quarter, exceeding all expectations.",
            "Market sentiment is bearish due to recent economic downturns.",
            "The stock price remained relatively stable throughout the trading day.",
            "Future outlook is uncertain, with mixed signals from various sectors.",
            "This is a great achievement for the team!",
            "Unfortunately, the project failed to meet its targets."
        ]

        def run_test(analyzer_instance, test_name):
            logger.info(f"--- Starting test: {test_name} ---")
            try:
                sentiment_results = analyzer_instance.predict_sentiment(sample_texts)
                if sentiment_results:
                    logger.info(f"Sentiment prediction successful for {test_name}.")
                    for i, text in enumerate(sample_texts):
                        print(f"\nText: {text}")
                        res = sentiment_results[i]
                        print(f"Sentiment: {res['label']}, Score: {res['score']:.4f}")
                        if 'raw_generated' in res:
                            print(f"Raw Generated: {res['raw_generated']}")
                else:
                    logger.error(f"Sentiment prediction failed or returned empty for {test_name}.")
            except Exception as e:
                logger.error(f"An error occurred during {test_name}: {e}", exc_info=True)
                print(f"{test_name} failed: {e}")
            logger.info(f"--- Finished test: {test_name} ---")

        # Test 1: Standard Classification Model (e.g., FinBERT)
        try:
            logger.info("Attempting to initialize with default FinBERT (classification)...")
            # Using a known small classifier to ensure example runs without large downloads if possible
            # Using ProsusAI/finbert as it was the original default
            analyzer_finbert = FinGPTSentimentAnalyzerService(
                base_model_name_or_path="ProsusAI/finbert",
                model_type="classification"
            )
            run_test(analyzer_finbert, "FinBERT Classification")
        except Exception as e:
            logger.error(f"Could not run FinBERT example: {e}. This might be due to model availability or setup.")
            print(f"FinBERT example failed to initialize or run: {e}")


        # Test 2: Generative Model with LoRA (Example - requires a base model and a LoRA adapter)
        # NOTE: For this example to run, you need a compatible base model (e.g., a small generative one)
        # and a LoRA adapter fine-tuned for sentiment on that base.
        # Replace "path/to/your/base_generative_model" and "path/to/your/lora_adapter"
        # with actual paths if you have them.
        # For demonstration, we'll use placeholder paths and expect it to fail gracefully if they don't exist.
        
        # A small generative model for testing (e.g. 'distilgpt2' if it can be adapted, or a small ChatGLM variant if available)
        # Using 'gpt2' as a widely available small generative model for placeholder.
        # A real scenario would use something like 'THUDM/chatglm2-6b'.
        base_generative_model_path = "gpt2" # Placeholder, a real ChatGLM-like model is better
        # Dummy LoRA path - this will likely fail unless you have a LoRA adapter for gpt2 sentiment
        dummy_lora_adapter_path = "path/to/dummy_lora_for_gpt2_sentiment"

        # Create dummy LoRA adapter files for the example to proceed further in init, if path doesn't exist
        # This is just to allow the example to attempt loading, it won't make a functional LoRA model.
        if not os.path.exists(dummy_lora_adapter_path):
            try:
                os.makedirs(dummy_lora_adapter_path, exist_ok=True)
                # Create minimal dummy files PEFT might look for
                with open(os.path.join(dummy_lora_adapter_path, "adapter_config.json"), "w") as f:
                    # A very minimal config. Real one is more complex.
                    f.write('{"base_model_name_or_path": "gpt2", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 32, "lora_dropout": 0.1}')
                with open(os.path.join(dummy_lora_adapter_path, "adapter_model.bin"), "w") as f:
                    f.write("dummy content") # Dummy weights file
                logger.info(f"Created dummy LoRA adapter files at {dummy_lora_adapter_path} for example run.")
                created_dummy_lora = True
            except Exception as e_dummy:
                logger.warning(f"Could not create dummy LoRA files: {e_dummy}")
                created_dummy_lora = False
        else:
            created_dummy_lora = False


        try:
            logger.info(f"Attempting to initialize with generative model '{base_generative_model_path}' and LoRA '{dummy_lora_adapter_path}'...")
            analyzer_generative_lora = FinGPTSentimentAnalyzerService(
                base_model_name_or_path=base_generative_model_path,
                lora_adapter_path=dummy_lora_adapter_path, # This will only work if path is valid & compatible
                model_type="generative",
                # load_in_4bit=True, # Enable if you have CUDA and want to test quantization
                # instruction_template=DEFAULT_INSTRUCTION_TEMPLATE (already default)
            )
            # This test will likely show "unknown" if dummy LoRA is used, or fail if model loading fails.
            run_test(analyzer_generative_lora, f"Generative ({base_generative_model_path}) with LoRA ({dummy_lora_adapter_path})")
        except Exception as e:
            logger.error(f"Could not run generative LoRA example: {e}. Ensure base model and LoRA adapter path are correct and compatible.")
            print(f"Generative LoRA example failed to initialize or run: {e}")
        finally:
            # Clean up dummy LoRA files if created
            if created_dummy_lora and os.path.exists(dummy_lora_adapter_path):
                try:
                    import shutil
                    shutil.rmtree(dummy_lora_adapter_path)
                    logger.info(f"Removed dummy LoRA adapter files from {dummy_lora_adapter_path}.")
                except Exception as e_clean:
                    logger.warning(f"Could not remove dummy LoRA files: {e_clean}")
            
            # Clean up dummy config from original example if it exists
            if os.path.exists("dummy_model_config.yaml"):
                os.remove("dummy_model_config.yaml")