# src/core/config.py

"""Configuration management with YAML and environment variables."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import AppConfig

class ConfigManager:
    """Manages loading of configuration from YAML files and environment variables."""

    def __init__(self):
        """Initializes the ConfigManager."""
        # This assumes the script is run from the project root.
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[AppConfig] = None
        self._prompts: Optional[Dict[str, str]] = None

    def get_config(self) -> AppConfig:
        """Get the application configuration, loading it if it hasn't been loaded."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> AppConfig:
        """Loads configuration from settings.yaml and environment variables."""
        # Start with default values from the AppConfig dataclass
        config = AppConfig()
        
        # Override with settings from config/settings.yaml
        settings_file = self.config_dir / "settings.yaml"
        if settings_file.exists():
            yaml_settings = self._load_yaml_file(settings_file)
            
            # Map YAML keys to AppConfig attributes
            field_mappings = {
                'model_name': 'model_name',
                'embedding_model': 'embedding_model',
                'language': 'language',
                'vector_db_path': 'vector_db_path',
                'collection_name': 'collection_name',
                'retrieval_k': 'retrieval_k',
                'data_dir': 'data_dir',
                'audio_dir': 'audio_dir',
                'transcripts_dir': 'transcripts_dir',
                'language_voice_map': 'language_voice_map' # ✨ NEW
            }
            
            for yaml_key, config_attr in field_mappings.items():
                if yaml_key in yaml_settings:
                    setattr(config, config_attr, yaml_settings[yaml_key])
        
        # Override with environment variables (highest priority)
        config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        config.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        return config

    def get_prompts(self) -> Dict[str, str]:
        """Get prompt templates from the prompts.yaml file."""
        if self._prompts is None:
            prompts_file = self.config_dir / "prompts.yaml"
            if prompts_file.exists():
                self._prompts = self._load_yaml_file(prompts_file)
            else:
                self._prompts = {
                    # Provide a default prompt if the file is missing
                    "multilingual_qa_prompt": """Use the following pieces of context from a video transcript to answer the question. If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}
IMPORTANT: You MUST provide the answer in the following language: {language}"""
                }
        return self._prompts

    def _load_yaml_file(self, file_path: Path) -> Dict:
        """Helper function to load a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content or {}
        except Exception as e:
            print(f"Warning: Could not load YAML file {file_path}. Error: {e}")
            return {}

# --- Global Singleton Instance and Helper Functions ---

_config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Global accessor for application configuration."""
    return _config_manager.get_config()

def get_prompts() -> Dict[str, str]:
    """Global accessor for prompt templates."""
    return _config_manager.get_prompts()

def validate_config() -> bool:
    """Global helper to validate the application configuration."""
    config = get_config()
    return config.validate()