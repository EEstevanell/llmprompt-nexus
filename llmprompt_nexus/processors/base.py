from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Union
import asyncio

from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig

logger = get_logger(__name__)

class BatchVariableProvider(Protocol):
    """Protocol defining how batch variables should be provided."""
    
    def get_global_variables(self) -> Dict[str, Any]:
        """Get variables that apply to all examples in the batch."""
        ...
        
    def get_example_variables(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Get variables specific to a single example in the batch."""
        ...

class BaseProcessor(ABC):
    """Base class for model processors."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, template: Optional[Template] = None):
        self.client = client
        self.model_config = model_config
        self.template = template
        logger.debug(f"Initialized processor for model {model_config.name}")
    
    @abstractmethod
    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        pass
    
    async def process_batch(self, items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple items."""
        results = []
        for item in items:
            result = await self.process_item(item)
            results.append(result)
        return results

    def _prepare_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """Prepare prompt from item and template.
        
        Args:
            item: Input data for the prompt
            
        Returns:
            Either a string prompt or a list of message dictionaries
            
        Raises:
            ValueError: If required data is missing
        """
        if self.template:
            # If we have a template, use its message formatting
            messages = []
            
            # Add system message if template provides one
            if self.template.system_message:
                messages.append({
                    "role": "system",
                    "content": self.template.system_message
                })
            
            # Add the rendered template as user message
            messages.append({
                "role": "user",
                "content": self.template.render(item)
            })
            
            return messages
            
        elif isinstance(item.get('messages'), list):
            # Handle pre-formatted messages
            return item['messages']
            
        elif isinstance(item.get('prompt'), str):
            # Handle raw prompt string
            messages = []
            
            # Add system message if provided in item
            if 'system_message' in item:
                messages.append({
                    "role": "system",
                    "content": item['system_message']
                })
            
            # Add user message
            messages.append({
                    "role": "user",
                    "content": item['prompt']
                })
            
            return messages
            
        else:
            raise ValueError("Item must contain either 'prompt', 'messages', or use a template")

    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process result. Override in subclasses if needed."""
        return result

class SimpleBatchVariableProvider:
    """A simple implementation of BatchVariableProvider with fixed global variables."""
    
    def __init__(self, global_vars: Dict[str, Any], variable_mapping: Optional[Dict[str, str]] = None):
        self.global_vars = global_vars
        self.variable_mapping = variable_mapping or {}
    
    def get_global_variables(self) -> Dict[str, Any]:
        return self.global_vars
    
    def get_example_variables(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if self.variable_mapping:
            return {
                self.variable_mapping.get(k, k): v 
                for k, v in example.items()
            }
        return example
