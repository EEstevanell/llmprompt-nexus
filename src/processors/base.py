from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Callable
import asyncio

from src.templates.base import Template
from src.utils.logger import get_logger
from src.clients.base import BaseClient

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
    
    def __init__(self, client: BaseClient):
        self.client = client
    
    @abstractmethod
    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        pass
    
    async def process_batch(self, items: List[Dict[str, Any]], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple items in parallel with rate limiting.
        
        This implementation leverages the client's built-in parallel processing.
        Subclasses can override this to provide custom batch processing logic.
        """
        if not items:
            return []
        
        # Convert items to prompts
        prompts = [self._prepare_prompt(item) for item in items]
        
        # Use client's parallel batch processing
        results = await self.client.generate_batch(prompts, model, **kwargs)
        
        # Post-process results if needed
        return [self._post_process_result(result) for result in results]
    
    def _prepare_prompt(self, item: Dict[str, Any]) -> str:
        """Prepare prompt from item. Override in subclasses if needed."""
        return str(item.get('prompt', ''))
    
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
