from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
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
    
    async def process_batch(self, 
                          items: List[Dict[str, Any]], 
                          global_vars: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[Callable] = None,
                          max_retries: int = 5,
                          max_concurrent: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process a batch of items with automatic queue management and rate limiting.
        
        Args:
            items: List of input items to process
            global_vars: Optional global variables to pass to all items
            progress_callback: Optional callback for progress updates
            max_retries: Maximum number of retries for failed requests
            max_concurrent: Maximum number of concurrent requests. If None, uses rate limit.
        """
        if not items:
            return []

        # Get rate limiter
        rate_limiter = self.client.get_rate_limiter(self.model_config.name)
        
        # Use rate limit as max concurrent if not specified
        if max_concurrent is None:
            max_concurrent = min(rate_limiter.max_calls, 10)  # Cap at 10 to prevent overwhelming
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(items)
        failed_indices = []

        async def process_item_with_retry(index: int, item: Dict[str, Any]) -> None:
            retry_count = 0
            base_delay = 1.0

            while retry_count <= max_retries:
                try:
                    async with semaphore:
                        result = await self.process_item(item, global_vars)
                        results[index] = result
                        if progress_callback:
                            progress_callback(index=index, status="completed")
                        return
                except Exception as e:
                    retry_count += 1
                    if "429" in str(e):
                        # Record rate limit failure
                        rate_limiter.record_failure(is_429=True)
                        # Use retry-after if provided, otherwise exponential backoff
                        delay = base_delay * (2 ** retry_count)
                        logger.warning(f"Rate limit hit, retry {retry_count}/{max_retries} in {delay}s")
                    else:
                        rate_limiter.record_failure()
                        delay = base_delay * (2 ** retry_count)
                        logger.error(f"Error processing item {index}: {str(e)}")

                    if retry_count <= max_retries:
                        await asyncio.sleep(delay)
                    else:
                        # Max retries exceeded
                        results[index] = {"error": str(e), "status": "failed"}
                        failed_indices.append(index)
                        if progress_callback:
                            progress_callback(index=index, status="failed")

        # Create and run tasks for all items
        tasks = [process_item_with_retry(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks)

        # Log summary of failures if any
        if failed_indices:
            logger.warning(f"Batch processing completed with {len(failed_indices)} failures")
            for idx in failed_indices:
                logger.debug(f"Failed item {idx}: {results[idx]['error']}")

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
