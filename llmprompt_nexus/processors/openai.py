from typing import Dict, List, Any, Optional, Union

from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIProcessor(BaseProcessor):
    """Processor for OpenAI models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        super().__init__(client, model_config, template)
        logger.debug(f"Initialized OpenAI processor for model {model_config.name}")
        
    def _prepare_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """Extract or format the prompt from the item."""
        if self.template:
            # If we have a template, use its message formatting
            return self.template.get_messages(item)
        elif isinstance(item.get('messages'), list):
            # Handle pre-formatted messages
            return item['messages']
        elif isinstance(item.get('prompt'), str):
            # Handle raw prompt string - wrap in messages format
            system_msg = item.get('system_message')
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": item['prompt']})
            return messages
        else:
            raise ValueError("Item must contain either 'prompt', 'messages', or use a template")

    async def process_item(self, item: Dict[str, Any], global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single item using OpenAI's API."""
        try:
            # Merge global vars with item if provided
            if global_vars:
                item = {**global_vars, **item}  # Item values take precedence

            # Get prompt or messages from template or input
            prompt = self._prepare_prompt(item)
            model = self.model_config.name
            
            # Only pass through OpenAI-specific parameters
            allowed_params = {
                'temperature', 'top_p', 'n', 'stream', 'stop', 
                'max_tokens', 'presence_penalty', 'frequency_penalty',
                'logit_bias', 'user'
            }
            kwargs = {k: v for k, v in item.items() 
                     if k in allowed_params}
            
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

    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the API response."""
        if isinstance(result, dict) and 'error' in result:
            return result
        
        return {
            'response': result.get('response', ''),
            'model': self.model_config.name,
            'usage': result.get('usage', {})
        }
