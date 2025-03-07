from abc import ABC, abstractmethod
from typing import Dict, List, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseProcessor(ABC):
    """Base class for model processors."""
    
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
            template = self.templates.get('template', '')
            if not template:
                logger.warning("No template found in templates dict")
                return str(item.get('text', ''))
                
            text = item.get('text', '')
            prompt = f"{template}\n{text}"
            logger.debug(f"Formatted prompt with template")
            return prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise
