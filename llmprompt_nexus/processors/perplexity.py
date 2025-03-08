from typing import Dict, List, Any, Optional, Union
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        self.client = client
        self.model_config = model_config
        self.template = template
        logger.debug(f"Initialized Perplexity processor for model {model_config.name}")
    
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
    
    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item using Perplexity's API."""
        try:
            # Get prompt or messages from template or input
            prompt = self._prepare_prompt(item)
            model = self.model_config.name
            
            # Pass through any additional parameters except those we handle
            kwargs = {k: v for k, v in item.items() 
                     if k not in ('prompt', 'model', 'messages', 'system_message')}
            
            # If prompt is a list of messages, pass directly
            if isinstance(prompt, list):
                result = await self.client.generate(
                    prompt=prompt,
                    model=model,
                    **kwargs
                )
            else:
                # For string prompts, pass system message separately if available
                system_message = None
                if self.template and self.template.system_message:
                    system_message = self.template.system_message
                elif 'system_message' in item:
                    system_message = item['system_message']
                
                result = await self.client.generate(
                    prompt=prompt,
                    model=model,
                    system_message=system_message,
                    **kwargs
                )
            
            return self._post_process_result(result)
            
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            return {
                "error": str(e),
                "model": self.model_config.name
            }
    
    async def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple items sequentially (Perplexity doesn't support batch processing)."""
        results = []
        for item in items:
            result = await self.process_item(item)
            results.append(result)
        return results
