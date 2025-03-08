from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
import httpx

from llmprompt_nexus.models.registry import registry
from llmprompt_nexus.rate_limiting.limiter import RateLimiter
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class BaseClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
    def get_rate_limiter(self, model: str) -> RateLimiter:
        """Get or create a rate limiter for a specific model."""
        if model in self.rate_limiters:
            return self.rate_limiters[model]
        
        try:
            model_config = registry.get_model(model)
            rate_limits = model_config.rate_limits or {}
            rpm = rate_limits.get("rpm", 10)  # Default to 10 RPM if not specified
            
            logger.info(f"Creating rate limiter for {model} with {rpm} requests per minute")
            
            self.rate_limiters[model] = RateLimiter(
                max_calls=rpm,
                period=60  # 60 seconds = 1 minute
            )
            
            return self.rate_limiters[model]
            
        except ValueError:
            logger.warning(f"Model {model} not found in registry, using default rate limits")
            self.rate_limiters[model] = RateLimiter(max_calls=5, period=60)
            return self.rate_limiters[model]
    
    async def check_rate_limit(self, model: str):
        """Check rate limits before making an API call."""
        limiter = self.get_rate_limiter(model)
        await limiter.acquire()
        
        # Log current usage
        usage = limiter.get_current_usage()
        logger.debug(f"Rate limit usage for {model}: {usage}")
    
    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt."""
        pass
    
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts in parallel with rate limiting.
        
        This default implementation processes prompts in parallel while respecting rate limits.
        Subclasses can override this to provide more efficient batch processing.
        """
        if not prompts:
            return []
            
        # Get model config and rate limits
        model_config = registry.get_model(model)
        rate_limiter = self.get_rate_limiter(model)
        
        # Create semaphore to limit concurrent requests based on rate limits
        # Use rate limit as max concurrent to maximize throughput while staying within limits
        max_concurrent = rate_limiter.max_calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    await self.check_rate_limit(model)
                    return await self.generate(prompt, model, **kwargs)
                except Exception as e:
                    logger.error(f"Error processing prompt in batch: {str(e)}")
                    return {"error": str(e), "model": model}
        
        # Create tasks for all prompts
        tasks = [process_prompt(prompt) for prompt in prompts]
        
        # Execute all tasks concurrently with rate limiting
        logger.info(f"Processing batch of {len(prompts)} prompts with max {max_concurrent} concurrent requests")
        results = await asyncio.gather(*tasks)
        
        return results
