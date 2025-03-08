from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from src.templates.base import Template
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
            formatted = self.template.format(**item)
            logger.debug("Formatted prompt using template")
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise
