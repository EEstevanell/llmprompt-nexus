from typing import Dict, List, Any, Optional, Union
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models with improved rate limit handling."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        self.client = client
        self.model_config = model_config
        self.template = template
        self._max_retries = 5
        self._max_concurrent = None
        self._batch_size = None
        logger.debug(f"Initialized Perplexity processor for model {model_config.name}")
    
    def configure(self, max_retries: int = 5, max_concurrent: Optional[int] = None, 
                 batch_size: Optional[int] = None) -> None:
        """Configure processing parameters."""
        self._max_retries = max_retries
        self._max_concurrent = max_concurrent
        self._batch_size = batch_size
    
    def _prepare_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """Extract or format the prompt from the item."""
        if self.template:
            # If we have a template, use its message formatting
            messages = self.template.get_messages(item)
            if isinstance(messages, list):
                return messages[-1]['content']  # Get the last message content as prompt
            return messages
        elif isinstance(item.get('messages'), list):
            # Handle pre-formatted messages - get the last user message
            messages = item['messages']
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    return msg['content']
            return messages[0]['content']  # Fallback to first message
        elif isinstance(item.get('prompt'), str):
            return item['prompt']
        else:
            raise ValueError("Item must contain either 'prompt', 'messages', or use a template")
    
    async def process_item(self, item: Dict[str, Any], 
                         global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single translation item with retry handling."""
        try:
            # Merge global variables if provided
            if global_vars:
                context = {**item, **global_vars}
            else:
                context = item.copy()

            # Get formatted messages
            messages = self._prepare_prompt(context)
            
            # Generate with built-in retry
            result = await self.client.generate(
                prompt=messages,
                model=self.model_config.name
            )
            
            # Extract the translation from the response
            if isinstance(result, dict) and 'choices' in result:
                translation = result['choices'][0]['message']['content']
            else:
                translation = str(result)
            
            return {
                'response': translation,
                'status': 'completed',
                'model': self.model_config.name
            }
            
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed',
                'model': self.model_config.name
            }
    
    async def process_batch(self, items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Process a batch of items with improved rate limit handling."""
        # Update processing config if provided in kwargs
        if 'max_retries' in kwargs:
            self._max_retries = kwargs['max_retries']
        if 'max_concurrent' in kwargs:
            self._max_concurrent = kwargs['max_concurrent']
            
        # Use parent's implementation with our config
        return await super().process_batch(
            items,
            max_retries=self._max_retries,
            max_concurrent=self._max_concurrent,
            **kwargs
        )
