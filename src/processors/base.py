from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Callable
from src.templates.base import Template
from src.utils.logger import get_logger

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
    
    def __init__(self, template: Optional[Template] = None):
        """Initialize processor with optional template."""
        self.template = template
    
    @abstractmethod
    async def process_item(self, item: Dict) -> Dict[str, Any]:
        """Process a single item."""
        pass
        
    @abstractmethod
    async def process_batch(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple items as a batch."""
        pass
        
    def _format_prompt(self, item: Dict) -> str:
        """Format the prompt using the template and item data."""
        try:
            if not self.template:
                logger.warning("No template provided, using raw text input")
                return str(item.get('text', ''))
            
            # Let the template handle the formatting with all available variables
            formatted = self.template.render(item)
            logger.debug("Formatted prompt using template")
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise

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
