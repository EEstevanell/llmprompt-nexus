from typing import Dict, List, Any, Optional
from src.clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base import BaseProcessor
from src.templates.base import Template
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        super().__init__(template)
        self.client = client
        self.model_config = model_config
        logger.debug(f"Initialized Perplexity processor for model {model_config.id}")
        
    async def process_item(self, item: Dict) -> Dict[str, Any]:
        """Process a single item."""
        try:
            prompt = self._format_prompt(item)
            logger.debug(f"Sending request to Perplexity API using model {self.model_config.model_name}")
            
            # Extract parameters from model config
            params = {}
            if self.model_config.parameters:
                params = self.model_config.parameters.copy()
            
            # Set default temperature if not specified
            if "temperature" not in params:
                params["temperature"] = 0.7
                
            # Apply any other parameters that might be specific to this request
            if "top_p" not in params and self.model_config.parameters and "top_p" in self.model_config.parameters:
                params["top_p"] = self.model_config.parameters["top_p"]
            
            response = await self.client.generate(
                prompt=prompt,
                model=self.model_config.model_name,
                **params
            )
            
            logger.debug("Received response from Perplexity API")
            return {'response': response.get('text', ''), 'model': self.model_config.id}
        except Exception as e:
            logger.error(f"Error in Perplexity process_item: {str(e)}")
            raise
        
    async def process_batch(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple items as a batch."""
        logger.warning("Batch processing not supported for Perplexity API, falling back to sequential processing")
        try:
            results = []
            for item in items:
                result = await self.process_item(item)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in Perplexity process_batch: {str(e)}")
            raise
