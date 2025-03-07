from typing import Dict

from clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base_processor import BaseProcessor
from src.processors.openai_processor import OpenAIProcessor
from src.processors.perplexity_processor import PerplexityProcessor

class ProcessorFactory:
    """Factory for creating processors."""
    
    def get_processor(
        self, 
        api_type: str, 
        client: BaseClient, 
        model_config: ModelConfig,
        templates: Dict
    ) -> BaseProcessor:
        """Get the appropriate processor for the API type."""
        if api_type == "openai":
            return OpenAIProcessor(client, model_config, templates)
        elif api_type == "perplexity":
            return PerplexityProcessor(client, model_config, templates)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
