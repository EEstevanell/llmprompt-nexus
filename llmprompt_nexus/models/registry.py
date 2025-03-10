# src/models/registry.py
import os
import yaml
from typing import Dict, Optional, Any
from pathlib import Path

from llmprompt_nexus.models.config import ModelConfig
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """Registry for managing model configurations."""
    
    def __init__(self):
        """Initialize an empty model registry."""
        self._models: Dict[str, ModelConfig] = {}
        self._is_loaded = False
        
    def load_models(self) -> None:
        """Load all model configurations from the config directory."""
        if self._is_loaded:
            return
            
        config_dir = Path(__file__).parent.parent / 'config' / 'models'
        
        if not config_dir.exists():
            raise FileNotFoundError(f"Model config directory not found: {config_dir}")
        
        logger.debug(f"Loading model configurations from {config_dir}")
            
        # Load each YAML file in the models directory
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
                # Each YAML file contains a list of models under 'models' key
                if 'models' not in config_data:
                    logger.warning(f"No models found in {config_file}")
                    continue
                    
                loaded_count = 0
                for model_config in config_data['models']:
                    if 'name' not in model_config:
                        logger.warning(f"Model config missing name in {config_file}")
                        continue
                        
                    model_id = model_config['name']
                    # provider is already in the model config
                    self._models[model_id] = ModelConfig(**model_config)
                    loaded_count += 1
                
                logger.debug(f"Loaded {loaded_count} models from {config_file.name}")
                    
        self._is_loaded = True
        logger.debug(f"Model registry initialized with {len(self._models)} models")
    
    def get_model(self, model_id: str) -> ModelConfig:
        """
        Get a model configuration by ID.
        
        Args:
            model_id: The ID of the model to retrieve
            
        Returns:
            ModelConfig: The model configuration
            
        Raises:
            ValueError: If the model ID is not found
        """
        if not self._is_loaded:
            self.load_models()
            
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found in registry")
            
        return self._models[model_id]
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """
        Get a dictionary of all available models.
        
        Returns:
            Dict[str, ModelConfig]: Dictionary mapping model IDs to their configurations
        """
        if not self._is_loaded:
            self.load_models()
            
        return self._models.copy()
        
    def clear(self) -> None:
        """Clear all loaded models from the registry."""
        self._models.clear()
        self._is_loaded = False
        logger.debug("Model registry cleared")

# Global registry instance
registry = ModelRegistry()