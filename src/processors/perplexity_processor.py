from typing import Dict, List, Any

from clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base_processor import BaseProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, templates: Dict):
        self.client = client
        self.model_config = model_config
        self.templates = templates
        logger.debug(f"Initialized Perplexity processor for model {model_config.id}")
        
    async def process_item(self, item: Dict) -> Dict[str, Any]:
        """Process a single item."""
        try:
            prompt = self._format_prompt(item)
            logger.debug("Sending request to Perplexity API")
            response = await self.client.generate(
                prompt=prompt,
                model=self.model_config.model_name,
                temperature=self.model_config.temperature
            )
            logger.debug("Received response from Perplexity API")
            return response
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
            
    def _format_prompt(self, item: Dict) -> str:
        """Format the prompt using the template and item data."""
        template = self.templates.get(item.get("intention", "default"), "")
        # Replace placeholders in the template with item data
        prompt = template
        for key, value in item.items():
            if isinstance(value, str):
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, value)
        return prompt
