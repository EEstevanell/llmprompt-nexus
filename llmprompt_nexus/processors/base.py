from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import asyncio
import math

from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger
from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.models.model_config import ModelConfig

logger = get_logger(__name__)

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
    
    async def process_batch(self, items: List[Dict[str, Any]], global_vars: Optional[Dict[str, Any]] = None, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple items with automatic batching based on rate limits.
        
        Args:
            items: List of items to process
            global_vars: Optional dictionary of variables that apply to all items in the batch
            **kwargs: Additional arguments passed to process_item
            
        Returns:
            List of processed items with results
        """
        if not items:
            return []
            
        # Get the rate limiter for this model
        rate_limiter = self.client.get_rate_limiter(self.model_config.name)
        usage = rate_limiter.get_current_usage()
        
        # Calculate available capacity and ideal concurrency
        available_slots = max(1, usage["max_calls"] - usage["calls"])
        
        # Process in sequential chunks if rate limited, otherwise process in parallel
        results = [None] * len(items)
        
        # Create task queue
        queue = asyncio.Queue()
        for i, item in enumerate(items):
            # Merge global variables with item variables
            if global_vars:
                merged_item = global_vars.copy()
                merged_item.update(item)
            else:
                merged_item = item
            await queue.put((i, merged_item))
            
        # Define worker function
        async def worker():
            while not queue.empty():
                try:
                    idx, item = await queue.get()
                    result = await self.process_item(item)
                    results[idx] = result
                    queue.task_done()
                except Exception as e:
                    logger.error(f"Error processing batch item: {str(e)}")
                    results[idx] = {"error": str(e), "model": self.model_config.name}
                    queue.task_done()
        
        # Determine number of workers based on rate limits
        # We want to avoid hitting rate limits, so we set concurrency
        # to be at most 80% of the available rate limit capacity
        concurrency = max(1, min(len(items), math.floor(available_slots * 0.8)))
        logger.info(f"Processing batch of {len(items)} items with {concurrency} concurrent workers")
        
        # Create and run workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        
        # Cancel any remaining worker tasks
        for w in workers:
            w.cancel()
        
        return results

    def _prepare_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """Prepare prompt from item and template."""
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
            
    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process result. Override in subclasses if needed."""
        return result
