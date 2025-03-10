"""Factory for creating model-specific processors."""

from typing import Optional, Dict, Any

from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.processors.perplexity import PerplexityProcessor
from llmprompt_nexus.processors.openai import OpenAIProcessor
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

def create_processor(
    client: BaseClient,
    model_config: ModelConfig,
    template: Optional[Template] = None,
    processing_config: Optional[Dict[str, Any]] = None
) -> BaseProcessor:
    """Create a processor instance based on model configuration.
    
    Args:
        client: API client instance
        model_config: Model configuration
        template: Optional template to use
        processing_config: Optional processing configuration including:
            - max_retries: Maximum number of retries (default: 5)
            - max_concurrent: Maximum concurrent requests (default: based on rate limits)
            - batch_size: Size of batches for processing (default: None)
    """
    processor_map = {
        "perplexity": PerplexityProcessor,
        "openai": OpenAIProcessor
    }
    
    # Get processor class based on provider
    processor_class = processor_map.get(model_config.provider)
    if not processor_class:
        raise ValueError(f"No processor available for provider: {model_config.provider}")
    
    # Create processor instance
    processor = processor_class(client, model_config, template)
    
    # Apply any processing configuration
    if processing_config:
        if hasattr(processor, 'configure'):
            processor.configure(**processing_config)
        else:
            logger.warning(f"Processor {processor_class.__name__} does not support configuration")
    
    return processor
