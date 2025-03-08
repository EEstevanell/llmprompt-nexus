from typing import Dict, Optional

from src.clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base import BaseProcessor
from src.processors.openai import OpenAIProcessor
from src.processors.perplexity import PerplexityProcessor
from src.templates.base import Template
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ProcessorFactory:
    """Factory for creating processors."""
    
    def get_processor(
        self, 
        api_type: str, 
        client: BaseClient, 
        model_config: ModelConfig,
        template: Optional[Template] = None
    ) -> BaseProcessor:
        """
        Get the appropriate processor for the API type.
        
        Args:
            api_type: The type of API (e.g., 'openai', 'perplexity')
            client: The API client instance
            model_config: Model configuration
            template: Optional Template instance for formatting prompts
        
        Returns:
            BaseProcessor: The appropriate processor instance
        
        Raises:
            ValueError: If the API type is not supported
        """
        logger.debug(f"Creating processor for API type: {api_type}")
        
        try:
            if api_type == "openai":
                processor = OpenAIProcessor(client, model_config, template)
                logger.info(f"Created OpenAI processor for model {model_config.id}")
                return processor
            elif api_type == "perplexity":
                processor = PerplexityProcessor(client, model_config, template)
                logger.info(f"Created Perplexity processor for model {model_config.id}")
                return processor
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
        except Exception as e:
            logger.error(f"Error creating processor for {api_type}: {str(e)}")
            raise
