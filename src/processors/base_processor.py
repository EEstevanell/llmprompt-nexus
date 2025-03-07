from abc import ABC, abstractmethod
from typing import Dict, List, Any

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
