# main.py

import asyncio
import os
from pathlib import Path
from typing import Dict, Optional, List

from src.core.framework import LLMFramework
from src.models.registry import registry
from src.templates.intention import templates
from src.utils.logger import get_logger, VerboseLevel
from src.models.config import ModelConfig

logger = get_logger(__name__)

def validate_api_keys(api_keys: Dict[str, Optional[str]]) -> None:
    """Validate that required API keys are present."""
    missing_keys = [key for key, value in api_keys.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

def validate_model_config(model_config: ModelConfig) -> None:
    """Validate model configuration."""
    if not model_config.name:
        raise ValueError("Model name is required")
    if not model_config.provider:
        raise ValueError("Model provider is required")
    if model_config.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")
    if model_config.provider not in ["openai", "perplexity"]:
        raise ValueError(f"Unsupported provider: {model_config.provider}")

def validate_models(models_to_run: List[str]) -> List[ModelConfig]:
    """Validate models and return their configurations."""
    model_configs = []
    for model_id in models_to_run:
        try:
            model_config = registry.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model '{model_id}' not found in registry")
            validate_model_config(model_config)
            model_configs.append(model_config)
        except Exception as e:
            logger.error(f"Error validating model {model_id}: {str(e)}")
            raise
    return model_configs

async def main():
    try:
        # Load API keys from environment variables
        api_keys = {
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY")
        }
        
        # Validate API keys
        validate_api_keys(api_keys)
        
        # Create framework instance
        framework = LLMFramework(api_keys)
        input_dir = Path("inputs")
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")
        
        # Select which models to run
        models_to_run = ["gpt-4", "sonar-small", "sonar-medium-online"]
        model_configs = validate_models(models_to_run)
        
        # Validate templates
        if not templates:
            raise ValueError("No templates found")
        
        input_files = list(input_dir.glob("*.tsv"))
        if not input_files:
            logger.warning(f"No .tsv files found in {input_dir}")
            return
            
        for model_config in model_configs:
            logger.info(f"Processing with model: {model_config.id}")
            
            # Determine if model supports batch processing
            supports_batch = model_config.api == "openai"
            batch_config = registry.get_batch_config("openai") if supports_batch else None
            
            batch_enabled = batch_config.enabled if batch_config else False
            batch_size = min(batch_config.max_requests_per_batch, 10) if batch_config else 10
                
            for file_path in input_files:
                logger.info(f"Processing {file_path} with model {model_config.id}")
                try:
                    await framework.process_file(
                        file_path=file_path,
                        model_config=model_config,
                        templates=templates,
                        batch_mode=batch_enabled,
                        batch_size=batch_size,
                        max_concurrent=5
                    )
                except Exception as e:
                    logger.error(f"Error processing {file_path} with {model_config.id}: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
