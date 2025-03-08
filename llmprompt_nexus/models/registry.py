# src/models/registry.py
import os
import yaml
from typing import Dict, Optional, Any
from pathlib import Path

from llmprompt_nexus.models.config import ModelConfig
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """Registry for model configurations."""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        logger.info("Initializing ModelRegistry")
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self.models[config.id] = config
        logger.info(f"Registered model: {config.id}")
    
    def get_model(self, model_id: str) -> ModelConfig:
        """Get a model configuration by ID."""
        logger.debug(f"Available models in registry: {list(self.models.keys())}")
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry. Available models: {list(self.models.keys())}")
        return self.models[model_id]
    
    def load_from_yaml(self, yaml_path: Path) -> None:
        """Load model configuration from YAML file."""
        logger.info(f"Loading model configurations from {yaml_path}")
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or not isinstance(config_data, dict):
                raise ValueError(f"Invalid YAML configuration in {yaml_path}")
            
            model_configs = config_data.get('models', [])
            if not model_configs:
                logger.warning(f"No model configurations found in {yaml_path}")
                return
            
            logger.info(f"Found {len(model_configs)} model configurations in {yaml_path}")
            for model_config in model_configs:
                try:
                    config = ModelConfig(**model_config)
                    self.register_model(config)
                except Exception as e:
                    logger.error(f"Error loading model config: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error loading YAML file {yaml_path}: {str(e)}")
            raise

# Global registry instance
registry = ModelRegistry()

# Load configurations from YAML files in config/models directory
config_dir = Path(__file__).parent.parent / 'config' / 'models'
if config_dir.exists():
    logger.info(f"Loading model configurations from directory: {config_dir}")
    for config_file in config_dir.glob('*.yaml'):
        registry.load_from_yaml(config_file)
else:
    logger.error(f"Model configuration directory not found: {config_dir}")