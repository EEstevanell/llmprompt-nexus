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
        self.models: Dict[str, ModelConfig] = {}
        self.batch_configs: Dict[str, BatchAPIConfig] = {}
        
        # Default to config/models if not specified
        config_dir = config_dir or os.path.join("config", "models")
        if os.path.exists(config_dir):
            self._load_configurations(config_dir)
        else:
            logger.warning(f"Configuration directory not found: {config_dir}")
    
    def _load_configurations(self, config_dir: str) -> None:
        """Load model configurations from YAML files."""
        logger.info(f"Loading model configurations from {config_dir}")
        
        try:
            for file_name in os.listdir(config_dir):
                if not file_name.endswith('.yaml'):
                    continue
                    
                file_path = os.path.join(config_dir, file_name)
                provider = os.path.splitext(file_name)[0]  # e.g., 'openai' from 'openai.yaml'
                
                try:
                    with open(file_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        
                    # Process models section
                    if "models" in config_data:
                        self._process_models_config(provider, config_data)
                        
                    # Process batch API configuration if present
                    if "batch_api" in config_data:
                        self._process_batch_config(provider, config_data["batch_api"])
                        
                    logger.info(f"Loaded configuration for provider: {provider}")
                    
                except Exception as e:
                    logger.error(f"Error loading configuration from {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error accessing configuration directory: {str(e)}")
            raise
    
    def _process_models_config(self, provider: str, config_data: Dict[str, Any]) -> None:
        """Process the models section of a configuration file."""
        models_data = config_data.get("models", {})
        if not isinstance(models_data, dict):
            logger.error(f"Models configuration for {provider} must be a dictionary")
            return
            
        for model_id, model_data in models_data.items():
            try:
                if not isinstance(model_data, dict):
                    logger.warning(f"Skipping invalid model data for {model_id} in {provider} config")
                    continue
                
                # Use model_id from the key if name is not specified
                name = model_data.get("name", model_id)
                
                # Create ModelConfig instance
                model_config = ModelConfig(
                    name=name,
                    provider=provider,
                    description=model_data.get("description", ""),
                    max_tokens=model_data.get("max_tokens", 4096),
                    rate_limits=model_data.get("rate_limits"),
                    parameters=model_data.get("parameters")
                )
                
                self.models[model_id] = model_config
                logger.debug(f"Registered model: {model_id} ({provider})")
                
            except Exception as e:
                logger.error(f"Error processing model configuration in {provider}: {str(e)}")
                continue
    
    def _process_batch_config(self, provider: str, batch_data: Dict[str, Any]) -> None:
        """Process batch API configuration."""
        try:
            batch_config = BatchAPIConfig(
                enabled=batch_data.get("enabled", False),
                max_requests_per_batch=batch_data.get("max_requests_per_batch", 50000),
                max_file_size_bytes=batch_data.get("max_file_size_bytes", 209715200),
                rate_limits=batch_data.get("rate_limits")
            )
            
            self.batch_configs[provider] = batch_config
            logger.debug(
                f"Registered batch config for {provider}: " +
                f"enabled={batch_config.enabled}, " +
                f"max_batch={batch_config.max_requests_per_batch}, " +
                f"max_size={batch_config.max_file_size_bytes/1024/1024:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch configuration for {provider}: {str(e)}")
    
    def get_model(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            ModelConfig instance
            
        Raises:
            ValueError: If model is not found
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            raise ValueError(f"Model '{model_id}' not found. Available models: {', '.join(self.models.keys())}")
            
        return self.models[model_id]
    
    def get_batch_config(self, provider: str) -> Optional[BatchAPIConfig]:
        """Get batch API configuration for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai')
            
        Returns:
            BatchAPIConfig if available, None otherwise
        """
        if provider not in self.batch_configs:
            logger.debug(f"No batch configuration found for provider: {provider}")
            return None
            
        return self.batch_configs[provider]
    
    def list_models(self, provider: Optional[str] = None) -> List[str]:
        """List available model IDs, optionally filtered by provider.
        
        Args:
            provider: Optional provider name to filter by
            
        Returns:
            List of model IDs
        """
        if provider:
            models = [
                model_id for model_id, config in self.models.items()
                if config.provider == provider
            ]
            logger.debug(f"Found {len(models)} models for provider {provider}")
            return models
        
        logger.debug(f"Found {len(self.models)} total models")
        return list(self.models.keys())
    
    def list_providers(self) -> List[str]:
        """List all available providers.
        
        Returns:
            List of provider names
        """
        providers = {config.provider for config in self.models.values()}
        logger.debug(f"Found {len(providers)} providers")
        return list(providers)
    
    def validate_configurations(self) -> List[str]:
        """Validate all loaded configurations.
        
        Returns:
            List of validation errors, empty if all valid
        """
        errors = []
        
        # Check for duplicate model IDs
        model_ids = {}
        for model_id, config in self.models.items():
            if model_id in model_ids:
                errors.append(
                    f"Duplicate model ID '{model_id}' found in providers "
                    f"'{model_ids[model_id]}' and '{config.provider}'"
                )
            model_ids[model_id] = config.provider
            
        # Validate each model configuration
        for model_id, config in self.models.items():
            # Required fields
            if not config.name:
                errors.append(f"Model '{model_id}' is missing name")
            if not config.provider:
                errors.append(f"Model '{model_id}' is missing provider")
                
            # Rate limits validation
            if config.rate_limits:
                for limit_type, value in config.rate_limits.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        errors.append(
                            f"Model '{model_id}' has invalid rate limit "
                            f"for {limit_type}: {value}"
                        )
            
            # Parameters validation
            if config.parameters:
                if not isinstance(config.parameters, dict):
                    errors.append(
                        f"Model '{model_id}' has invalid parameters type: "
                        f"{type(config.parameters)}"
                    )
                    
        # Validate batch configurations
        for provider, config in self.batch_configs.items():
            if config.max_requests_per_batch <= 0:
                errors.append(
                    f"Provider '{provider}' has invalid max_requests_per_batch: "
                    f"{config.max_requests_per_batch}"
                )
            if config.max_file_size_bytes <= 0:
                errors.append(
                    f"Provider '{provider}' has invalid max_file_size_bytes: "
                    f"{config.max_file_size_bytes}"
                )
                
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
        else:
            logger.info("All configurations validated successfully")
            
        return errors

# Create a global instance
registry = ModelRegistry()