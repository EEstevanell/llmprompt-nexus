from typing import Dict, List, Any, Optional
from src.clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base import BaseProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        super().__init__(template)
        self.client = client
        self.model_config = model_config
        logger.debug(f"Initialized Perplexity processor for model {model_config.id}")
        
    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item using Perplexity's API."""
        try:
            prompt = self._prepare_prompt(item)
            model = item.get('model', 'sonar')
            
            # Pass through any additional parameters
            kwargs = {k: v for k, v in item.items() 
                     if k not in ('prompt', 'model')}
            
            result = await self.client.generate(prompt, model, **kwargs)
            return self._post_process_result(result)
            
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            return {
                "error": str(e),
                "model": model
            }
    
    def _prepare_prompt(self, item: Dict[str, Any]) -> str:
        """Extract or format the prompt from the item."""
        if isinstance(item.get('prompt'), str):
            return item['prompt']
        elif isinstance(item.get('messages'), list):
            # Handle message-style prompts
            return item['messages'][-1]['content']
        else:
            raise ValueError("Item must contain either 'prompt' or 'messages'")
    
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
