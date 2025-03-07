# src/models/registry.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.models.config import ModelConfig, BatchAPIConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """Registry of available models loaded from configuration files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        self.models: Dict[str, ModelConfig] = {}
        self.batch_configs: Dict[str, BatchAPIConfig] = {}
        
        # Set default config directory if not provided
        if config_dir is None:
            # Try to find the config directory relative to the current script
            script_dir = Path(__file__).resolve().parent.parent.parent
            config_dir = os.path.join(script_dir, "config", "models")
        
        # Load configurations
        self._load_configurations(config_dir)
    
    def _load_configurations(self, config_dir: str) -> None:
        """Load model configurations from YAML files.
        
        Args:
            config_dir: Directory containing model configuration files
        """
        config_path = Path(config_dir)
        if not config_path.exists():
            logger.warning(f"Configuration directory not found: {config_dir}")
            return
        
        # Load all YAML files in the config directory
        yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
        
        if not yaml_files:
            logger.warning(f"No YAML configuration files found in {config_dir}")
            return
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract provider/API name
                provider = config_data.get("provider", yaml_file.stem)
                
                # Process models
                self._process_models_config(provider, config_data)
                
                # Process batch API config if available
                if "batch_api" in config_data and provider == "openai":
                    self._process_batch_config(provider, config_data["batch_api"])
                
                logger.info(f"Loaded configuration from {yaml_file.name}")
            except Exception as e:
                logger.error(f"Error loading configuration from {yaml_file}: {e}")
    
    def _process_models_config(self, provider: str, config_data: Dict[str, Any]) -> None:
        """Process and register models from configuration data.
        
        Args:
            provider: Provider name (openai, perplexity)
            config_data: Configuration data from YAML
        """
        if "models" not in config_data:
            logger.warning(f"No models defined in configuration for {provider}")
            return
        
        models_config = config_data["models"]
        for model_id, model_data in models_config.items():
            # For some providers, the API model name might be different from the model ID
            model_name = model_data.get("name", model_id)
            
            # Create model config
            model_config = ModelConfig(
                name=model_name,
                provider=provider,
                description=model_data.get("description", ""),
                max_tokens=model_data.get("max_tokens", 4096),
                rate_limits=model_data.get("rate_limits", {}),
                parameters=model_data.get("parameters", {})
            )
            
            # Register the model
            self.models[model_id] = model_config
    
    def _process_batch_config(self, provider: str, batch_data: Dict[str, Any]) -> None:
        """Process batch API configuration.
        
        Args:
            provider: Provider name
            batch_data: Batch API configuration from YAML
        """
        batch_config = BatchAPIConfig(
            enabled=batch_data.get("enabled", False),
            max_requests_per_batch=batch_data.get("max_requests_per_batch", 50000),
            max_file_size_bytes=batch_data.get("max_file_size_bytes", 209715200),
            rate_limits=batch_data.get("rate_limits", {})
        )
        
        self.batch_configs[provider] = batch_config
    
    def get_model(self, model_id: str) -> ModelConfig:
        """Get model config by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelConfig for the requested model
            
        Raises:
            ValueError: If model ID is not found
        """
        if model_id not in self.models:
            raise ValueError(f"Unknown model: {model_id}. Available models: {', '.join(self.models.keys())}")
        return self.models[model_id]
    
    def get_batch_config(self, provider: str) -> Optional[BatchAPIConfig]:
        """Get batch API configuration for a provider.
        
        Args:
            provider: Provider name (e.g., 'openai')
            
        Returns:
            BatchAPIConfig if available, None otherwise
        """
        return self.batch_configs.get(provider)
    
    def list_models(self, provider: Optional[str] = None) -> List[str]:
        """List available models, optionally filtered by provider.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            List of model IDs
        """
        if provider:
            return [
                model_id for model_id, model in self.models.items()
                if model.provider == provider
            ]
        return list(self.models.keys())
    
    def list_providers(self) -> List[str]:
        """List unique providers across all models.
        
        Returns:
            List of provider names
        """
        return list(set(model.provider for model in self.models.values()))

# Create a singleton instance
registry = ModelRegistry()