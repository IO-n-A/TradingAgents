# MLOps/experiment_tracking/mlflow_utils.py

import mlflow
import logging
import logging.config
import os
import sys
import yaml
import pandas as pd

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
# Assuming logging_config.py is in config/ at the project root
logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
if os.path.exists(logging_config_path):
    logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
else:
    # Basic config if file not found, to ensure logger works
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Class: MLflowTracker
# Description: A wrapper class to simplify interactions with MLflow for experiment tracking.
#              It handles setting the tracking URI, getting or creating experiments,
#              starting and ending runs, and logging parameters, metrics, artifacts, and models.
# Input:
#   tracking_uri: String, optional MLflow tracking server URI. Defaults to local './mlruns'.
#   experiment_name: String, optional name for the MLflow experiment. Defaults to "DefaultFinAIExperiment".
# Output: An instance of MLflowTracker.
# Dependencies: mlflow, logging, os.
class MLflowTracker:
    print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Purpose: Wrapper for MLflow tracking functionalities, Output: MLflowTracker instance.")
    """
    A wrapper class for MLflow tracking functionalities.
    """
    # Method: __init__
    # Description: Initializes the MLflowTracker, sets the tracking URI, and gets/creates the specified experiment.
    # Input:
    #   tracking_uri: String, optional MLflow tracking server URI.
    #   experiment_name: String, optional name for the MLflow experiment.
    # Output: None.
    # Dependencies: mlflow, logging, os.
    def __init__(self, tracking_uri: str = None, experiment_name: str = "DefaultFinAIExperiment"):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: __init__, Purpose: Initializes MLflow tracker, sets URI and experiment, Output: None.")
        """
        Initializes the MLflowTracker.
        The tracking URI is determined with the following precedence:
        1. MLFLOW_TRACKING_URI environment variable.
        2. 'tracking_uri' parameter (e.g., from a config file).
        3. Default local './mlruns' directory.

        Args:
            tracking_uri (str, optional): The MLflow tracking server URI, typically from a config file.
                                          This is used if MLFLOW_TRACKING_URI env var is not set.
            experiment_name (str, optional): The name of the MLflow experiment.
                                             Defaults to "DefaultFinAIExperiment".
        """
        env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        if env_tracking_uri:
            mlflow.set_tracking_uri(env_tracking_uri)
            logger.info(f"MLflow tracking URI set from MLFLOW_TRACKING_URI environment variable: {env_tracking_uri}")
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set from constructor argument: {tracking_uri}")
        else:
            # Default to local tracking if no URI is provided via env var or argument
            default_mlruns_path = os.path.join(project_root, "mlruns")
            if not os.path.exists(default_mlruns_path):
                os.makedirs(default_mlruns_path)
            # Ensure the path is a valid URI for local tracking
            local_uri = f"file:{os.path.abspath(default_mlruns_path)}"
            mlflow.set_tracking_uri(local_uri)
            logger.info(f"MLflow tracking URI set to default local: {local_uri}")

        self.experiment_name = experiment_name
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        self.active_run = None

    # Method: _get_or_create_experiment
    # Description: Retrieves an existing MLflow experiment by name or creates it if it doesn't exist.
    # Input:
    #   name: String, the name of the experiment.
    # Output: String, the ID of the experiment.
    # Dependencies: mlflow, logging.
    def _get_or_create_experiment(self, name: str) -> str:
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: _get_or_create_experiment, Purpose: Gets or creates an MLflow experiment, Output: Experiment ID string.")
        """Gets an existing experiment ID or creates a new one."""
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            logger.info(f"Experiment '{name}' not found. Creating new experiment.")
            experiment_id = mlflow.create_experiment(name)
            logger.info(f"Experiment '{name}' created with ID: {experiment_id}")
            return experiment_id
        else:
            logger.info(f"Experiment '{name}' found with ID: {experiment.experiment_id}")
            return experiment.experiment_id

    # Method: start_run
    # Description: Starts a new MLflow run within the configured experiment.
    #              If an active run already exists and `nested` is False, the existing run is ended first.
    # Input:
    #   run_name: String, optional name for the run.
    #   nested: Boolean, optional, True if this is a nested run (default False).
    # Output: mlflow.ActiveRun object.
    # Dependencies: mlflow, logging.
    def start_run(self, run_name: str = None, nested: bool = False):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: start_run, Purpose: Starts an MLflow run, Output: ActiveRun object.")
        """
        Starts an MLflow run.

        Args:
            run_name (str, optional): Name for the run. Defaults to None.
            nested (bool, optional): Whether this is a nested run. Defaults to False.

        Returns:
            mlflow.ActiveRun: The active MLflow run object.
        """
        if self.active_run and not nested:
            logger.warning(f"An active run '{self.active_run.info.run_name}' already exists. Ending it before starting a new one.")
            self.end_run()
        
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        )
        logger.info(f"Started MLflow run: {self.active_run.info.run_name} (ID: {self.active_run.info.run_id})")
        return self.active_run

    # Method: end_run
    # Description: Ends the currently active MLflow run.
    # Input: None.
    # Output: None.
    # Dependencies: mlflow, logging.
    def end_run(self):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: end_run, Purpose: Ends the current MLflow run, Output: None.")
        """Ends the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.active_run.info.run_name} (ID: {self.active_run.info.run_id})")
            self.active_run = None
        else:
            logger.warning("No active MLflow run to end.")

    # Method: log_param
    # Description: Logs a single parameter to the active MLflow run.
    # Input:
    #   key: String, the name of the parameter.
    #   value: Any, the value of the parameter.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_param(self, key: str, value: any):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_param, Purpose: Logs a single parameter to MLflow, Output: None.")
        """Logs a single parameter."""
        if self.active_run:
            mlflow.log_param(key, value)
            logger.debug(f"Logged param: {{'{key}': {value}}}")
        else:
            logger.error("Cannot log parameter: No active MLflow run.")

    # Method: log_params
    # Description: Logs multiple parameters from a dictionary to the active MLflow run.
    # Input:
    #   params_dict: Dictionary, parameters to log.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_params(self, params_dict: dict):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_params, Purpose: Logs multiple parameters to MLflow, Output: None.")
        """Logs multiple parameters from a dictionary."""
        if self.active_run:
            mlflow.log_params(params_dict)
            logger.info(f"Logged parameters: {params_dict}")
        else:
            logger.error("Cannot log parameters: No active MLflow run.")

    # Method: log_metric
    # Description: Logs a single metric to the active MLflow run.
    # Input:
    #   key: String, the name of the metric.
    #   value: Float, the value of the metric.
    #   step: Integer, optional step for the metric.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_metric(self, key: str, value: float, step: int = None):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_metric, Purpose: Logs a single metric to MLflow, Output: None.")
        """Logs a single metric."""
        if self.active_run:
            mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged metric: {{'{key}': {value}}} at step {step if step else 'N/A'}")
        else:
            logger.error("Cannot log metric: No active MLflow run.")

    # Method: log_metrics
    # Description: Logs multiple metrics from a dictionary to the active MLflow run.
    # Input:
    #   metrics_dict: Dictionary, metrics to log.
    #   step: Integer, optional step for the metrics.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_metrics(self, metrics_dict: dict, step: int = None):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_metrics, Purpose: Logs multiple metrics to MLflow, Output: None.")
        """Logs multiple metrics from a dictionary."""
        if self.active_run:
            mlflow.log_metrics(metrics_dict, step=step)
            logger.info(f"Logged metrics: {metrics_dict} at step {step if step else 'N/A'}")
        else:
            logger.error("Cannot log metrics: No active MLflow run.")

    # Method: log_artifact
    # Description: Logs a local file or directory as an artifact to the active MLflow run.
    # Input:
    #   local_path: String, path to the local file or directory.
    #   artifact_path: String, optional path within the MLflow run's artifact store.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_artifact(self, local_path: str, artifact_path: str = None):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_artifact, Purpose: Logs a local file/directory as an artifact to MLflow, Output: None.")
        """Logs a local file or directory as an artifact."""
        if self.active_run:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.info(f"Logged artifact: '{local_path}' to '{artifact_path if artifact_path else ''}'")
        else:
            logger.error("Cannot log artifact: No active MLflow run.")
            
    # Method: log_artifacts
    # Description: Logs all files in a local directory as artifacts to the active MLflow run.
    # Input:
    #   local_dir: String, path to the local directory.
    #   artifact_path: String, optional path within the MLflow run's artifact store.
    # Output: None.
    # Dependencies: mlflow, logging.
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_artifacts, Purpose: Logs all files in a directory as artifacts to MLflow, Output: None.")
        """Logs all files in a local directory as artifacts."""
        if self.active_run:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            logger.info(f"Logged artifacts from directory: '{local_dir}' to '{artifact_path if artifact_path else ''}'")
        else:
            logger.error("Cannot log artifacts: No active MLflow run.")

    # Method: log_model
    # Description: Logs a model to the active MLflow run. Attempts to use flavor-specific logging
    #              (e.g., for Stable Baselines3 via SB3Wrapper) and falls back to generic model logging.
    # Input:
    #   model: The model object to log.
    #   artifact_path: String, path within the MLflow run's artifact store for the model.
    #   registered_model_name: String, optional name to register the model under in MLflow Model Registry.
    #   **kwargs: Additional arguments for `mlflow.<flavor>.log_model`.
    # Output: None.
    # Dependencies: mlflow, logging, SB3Wrapper (local class).
    def log_model(self, model, artifact_path: str, registered_model_name: str = None, **kwargs):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: MLflowTracker, Function: log_model, Purpose: Logs a model to MLflow, attempting flavor-specific logging, Output: None.")
        """
        Logs a model. For scikit-learn compatible models, use mlflow.sklearn.log_model.
        For PyTorch, use mlflow.pytorch.log_model. For TensorFlow, mlflow.tensorflow.log_model.
        For generic Python functions, mlflow.pyfunc.log_model.
        This is a generic wrapper; specific model types might need specific mlflow.<flavor>.log_model calls.
        """
        if self.active_run:
            # Basic model logging (generic, may need flavor-specific for full features)
            # Example for a generic case, you might need to adapt based on model type
            try:
                if "stable_baselines3" in str(type(model)): # Basic check for SB3 models
                    mlflow.pyfunc.log_model(
                        artifact_path=artifact_path,
                        python_model=SB3Wrapper(model), # Requires a PyFunc wrapper for SB3
                        # signature=... # Optional: define model signature
                        registered_model_name=registered_model_name,
                        **kwargs
                    )
                    logger.info(f"Logged SB3 model to: {artifact_path}, registered as: {registered_model_name}")
                # Add other flavors like PyTorch, TensorFlow, etc.
                # elif isinstance(model, torch.nn.Module):
                #    mlflow.pytorch.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
                #    logger.info(f"Logged PyTorch model to: {artifact_path}, registered as: {registered_model_name}")
                else: # Fallback for other types or if specific logging fails
                    mlflow.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
                    logger.info(f"Logged generic model to: {artifact_path}, registered as: {registered_model_name}")
            except Exception as e:
                logger.error(f"Failed to log model using specific flavor, attempting generic log_model. Error: {e}")
                try:
                    # Fallback to generic model logging if specific flavor fails or is not identified
                    mlflow.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
                    logger.info(f"Logged generic model (fallback) to: {artifact_path}, registered as: {registered_model_name}")
                except Exception as final_e:
                    logger.error(f"Failed to log model even with generic mlflow.log_model: {final_e}", exc_info=True)
        else:
            logger.error("Cannot log model: No active MLflow run.")

# Wrapper for Stable Baselines3 models to be compatible with mlflow.pyfunc.log_model
# This is a simplified example. A more robust wrapper would handle environment creation,
# observation preprocessing, etc., if needed for inference.
# Class: SB3Wrapper
# Description: A wrapper class to make Stable Baselines3 (SB3) models compatible with `mlflow.pyfunc.log_model`.
#              It implements the `mlflow.pyfunc.PythonModel` interface.
# Input:
#   model: A trained Stable Baselines3 model object.
# Output: An instance of SB3Wrapper.
# Dependencies: mlflow.pyfunc.PythonModel, pandas (pd).
class SB3Wrapper(mlflow.pyfunc.PythonModel):
    print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: SB3Wrapper, Purpose: Wrapper for Stable Baselines3 models for MLflow PyFunc compatibility, Output: SB3Wrapper instance.")
    # Method: __init__
    # Description: Initializes the SB3Wrapper with an SB3 model.
    # Input:
    #   model: A trained Stable Baselines3 model object.
    # Output: None.
    # Dependencies: None.
    def __init__(self, model):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: SB3Wrapper, Function: __init__, Purpose: Initializes wrapper with an SB3 model, Output: None.")
        self.model = model

    # Method: predict
    # Description: Performs prediction using the wrapped SB3 model.
    #              The input `model_input` is expected to be observations compatible with the SB3 model.
    # Input:
    #   context: MLflow context object (not used in this simple wrapper).
    #   model_input: The input data for prediction (e.g., pandas DataFrame or numpy array of observations).
    # Output: A pandas DataFrame containing the predictions (actions).
    # Dependencies: pandas (pd).
    def predict(self, context, model_input):
        print(f"File: MLOps/experiment_tracking/mlflow_utils.py, Class: SB3Wrapper, Function: predict, Purpose: Performs prediction using the wrapped SB3 model, Output: DataFrame of actions.")
        # model_input is expected to be a pandas DataFrame or numpy array
        # This predict function needs to match how your SB3 model expects input for prediction
        # For SB3, it's typically an observation.
        # This is a very basic example; you might need to preprocess model_input
        # to match the observation space of the environment the model was trained on.
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values
        
        predictions = []
        for i in range(len(model_input)):
            obs = model_input[i]
            # Ensure observation is in the correct shape/format for your model
            # For SB3 PPO, model.predict() returns action and next hidden states (if LSTM)
            action, _states = self.model.predict(obs, deterministic=True)
            predictions.append(action)
        return pd.DataFrame(predictions)


# Example usage (can be removed or kept for testing the module)
if __name__ == "__main__":
    logger.info("Testing MLflowUtils...")
    
    # Load global config to get MLflow URI if defined
    global_config_path = os.path.join(project_root, "MLOps", "config", "common", "global_vars.yaml")
    mlflow_tracking_uri = None
    if os.path.exists(global_config_path):
        try:
            with open(global_config_path, 'r') as f:
                global_cfg = yaml.safe_load(f)
            mlflow_tracking_uri = global_cfg.get("mlflow_tracking_uri")
        except Exception as e:
            logger.warning(f"Could not load MLflow tracking URI from global_vars.yaml: {e}")

    tracker = MLflowTracker(tracking_uri=mlflow_tracking_uri, experiment_name="MLflowUtilsTestExperiment")
    
    with tracker.start_run(run_name="TestRun_MLflowUtils"):
        tracker.log_param("test_param_1", 123)
        tracker.log_params({"framework": "FinAI", "version": "0.1"})
        
        tracker.log_metric("accuracy", 0.95, step=1)
        tracker.log_metrics({"loss": 0.05, "val_accuracy": 0.92}, step=1)
        
        # Create a dummy artifact
        dummy_artifact_path = os.path.join(project_root, "MLOps", "experiment_tracking", "dummy_artifact.txt")
        with open(dummy_artifact_path, "w") as f:
            f.write("This is a test artifact.")
        tracker.log_artifact(dummy_artifact_path, "test_artifacts")
        os.remove(dummy_artifact_path) # Clean up

        # Placeholder for model logging example
        # class DummyModel:
        #     def predict(self, data): return [0]*len(data)
        # dummy_model = DummyModel()
        # tracker.log_model(dummy_model, "dummy_pyfunc_model", registered_model_name="TestDummyModel")

    logger.info("MLflowUtils test run completed. Check your MLflow UI.")