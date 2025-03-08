from typing import Dict, Optional
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.processors.openai import OpenAIProcessor
from llmprompt_nexus.processors.perplexity import PerplexityProcessor
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

def create_processor(
    client: BaseClient,
    model_config: ModelConfig,
    template: Optional[Template] = None
) -> BaseProcessor:
    """Create a processor instance based on the model provider.
    
    Args:
        client: The API client instance
        model_config: Configuration for the model
        template: Optional template for processing
        
    Returns:
        An instance of the appropriate processor class
        
    Raises:
        ValueError: If the model provider is not supported
    """
    processors = {
        "openai": OpenAIProcessor,
        "perplexity": PerplexityProcessor
    }
    
    provider = model_config.provider
    if provider not in processors:
        raise ValueError(f"Unsupported model provider: {provider}")
    
    processor_class = processors[provider]
    return processor_class(client, model_config, template)
