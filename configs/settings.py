import os
import logging # New import
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, PydanticBaseSettingsSource, YamlConfigSettingsSource
from typing import Optional, Type, Tuple, Dict, Any
from pathlib import Path

# Define the path to the config directory
CONFIG_DIR = Path(__file__).parent
API_KEYS_YAML_PATH = CONFIG_DIR / "api_keys.yaml"
logger = logging.getLogger(__name__) # New logger

# Manages application configuration by loading settings from YAML, environment variables, and .env files with a defined priority.
# It defines various API keys (e.g., ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY) and database settings (MONGODB_URI), inheriting from pydantic_settings.BaseSettings.
# Key dependencies include pydantic-settings for settings management, pathlib for path operations, and logging for status messages.
class AppConfig(BaseSettings):
    """
    Application configuration class using Pydantic-Settings.
    Loads settings primarily from api_keys.yaml, then environment variables, then .env file.
    """
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: Optional[str] = None
    REDDIT_USERNAME: Optional[str] = None
    REDDIT_PASSWORD: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET_KEY: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = None

    # Database Settings
    MONGODB_URI: Optional[str] = None
    MONGODB_DATABASE_NAME: Optional[str] = None

    # Initializes the AppConfig instance, triggering the settings loading mechanism via super().__init__ and pydantic-settings' customization.
    # It performs post-loading checks for the existence and content of the API_KEYS_YAML_PATH, logging warnings using the 'logger' instance if issues are found.
    # This method relies on the parent class initializer, pathlib for file checks, and the logging module.
    def __init__(self, **values: Any):
        print(f"START: Initializing AppConfig in {__name__} (settings.py)")
        # super().__init__ will trigger settings_customise_sources and load data
        super().__init__(**values)

        # Post-loading checks and print statements
        if not API_KEYS_YAML_PATH.exists():
            warning_msg = (
                f"API keys YAML file not found at {API_KEYS_YAML_PATH}. "
                "API key dependent features may not work if keys are not provided via other means (env, .env)."
            )
            logger.warning(warning_msg)
            print(f"ERROR: {warning_msg} (settings.py - AppConfig.__init__)")
        elif API_KEYS_YAML_PATH.read_text(encoding='utf-8').strip() == "":
            warning_msg = (
                f"API keys YAML file at {API_KEYS_YAML_PATH} is empty. "
                "API key dependent features may not work if keys are not provided via other means (env, .env)."
            )
            logger.warning(warning_msg)
            print(f"ERROR: {warning_msg} (settings.py - AppConfig.__init__)")
        else:
            print(f"INFO: API keys YAML file exists at {API_KEYS_YAML_PATH} and was considered as a configuration source. (settings.py - AppConfig.__init__)")

        print(f"SUCCESS: AppConfig initialized. Loaded configuration. (settings.py - AppConfig.__init__)")

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / '.env',  # Path to your .env file
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields from sources
        # secrets_dir='/var/run/secrets' # Example for Docker secrets
    )

    # Customizes the settings loading order for the AppConfig class, a pydantic-settings hook.
    # It prioritizes YamlConfigSettingsSource (using API_KEYS_YAML_PATH), followed by environment variables (env_settings), and then the .env file (dotenv_settings).
    # This classmethod primarily depends on various source classes from pydantic_settings.sources to define the configuration loading strategy.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource, # Use base class for type hint
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource, # Placeholder for file_secret_settings if used
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        print(f"INFO: Customizing configuration sources for AppConfig. Prioritizing YAML. (settings.py - settings_customise_sources)")
        # New order: YAML, then Env, then .env
        return (
            YamlConfigSettingsSource(settings_cls, yaml_file=API_KEYS_YAML_PATH), # Priority 1: YAML file
            env_settings,  # Priority 2: Environment variables
            dotenv_settings, # Priority 3: .env file
            init_settings,
            # file_secret_settings, # Not used for now
        )

# Instantiate AppConfig
app_config = AppConfig()