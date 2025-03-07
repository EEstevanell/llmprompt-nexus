from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from src.models.registry import registry
from src.rate_limiting.limiter import RateLimiter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
    def get_rate_limiter(self, model: str) -> RateLimiter:
        """Get or create a rate limiter for a specific model.
        
        Args:
            model: The model name/id
            
        Returns:
            A RateLimiter instance configured for the model
        """
        if model in self.rate_limiters:
            return self.rate_limiters[model]
        
        # Get model config from registry
        try:
            model_config = registry.get_model(model)
            rate_limits = model_config.rate_limits or {}
            
            # Create rate limiter based on RPM (requests per minute)
            rpm = rate_limits.get("rpm", 10)  # Default to 10 RPM if not specified
            
            logger.info(f"Creating rate limiter for {model} with {rpm} requests per minute")
            self.rate_limiters[model] = RateLimiter(rpm, 60)  # 60 seconds = 1 minute
            
            return self.rate_limiters[model]
        except ValueError:
            # If model not found in registry, create a conservative default limiter
            logger.warning(f"Model {model} not found in registry. Using default rate limiter.")
            self.rate_limiters[model] = RateLimiter(10, 60)  # Default to 10 RPM
            return self.rate_limiters[model]
        
    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt."""
        pass
        
    @abstractmethod
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts."""
        pass
