from typing import Dict

from clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base_processor import BaseProcessor
from src.processors.openai_processor import OpenAIProcessor
from src.processors.perplexity_processor import PerplexityProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
        logger.debug(f"Creating processor for API type: {api_type}")
        
        try:
            if api_type == "openai":
                processor = OpenAIProcessor(client, model_config, templates)
                logger.info(f"Created OpenAI processor for model {model_config.id}")
                return processor
            elif api_type == "perplexity":
                processor = PerplexityProcessor(client, model_config, templates)
                logger.info(f"Created Perplexity processor for model {model_config.id}")
                return processor
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
        except Exception as e:
            logger.error(f"Error creating processor for {api_type}: {str(e)}")
            raise
