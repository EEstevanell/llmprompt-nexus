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
            
            # Get RPM and TPM limits
            rpm = rate_limits.get("rpm", 10)  # Default to 10 RPM if not specified
            tpm = rate_limits.get("tpm")  # Token limit is optional
            
            logger.info(f"Creating rate limiter for {model} with {rpm} requests per minute" +
                       (f" and {tpm} tokens per minute" if tpm else ""))
            
            # Create rate limiter with both request and token limits
            self.rate_limiters[model] = RateLimiter(
                max_calls=rpm,
                period=60,  # 60 seconds = 1 minute
                max_tokens_per_min=tpm
            )
            
            return self.rate_limiters[model]
            
        except ValueError:
            logger.warning(f"Model {model} not found in registry, using default rate limits")
            # Use conservative defaults for unknown models
            self.rate_limiters[model] = RateLimiter(max_calls=5, period=60)
            return self.rate_limiters[model]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        This is a very rough estimation. Subclasses should override this
        with more accurate model-specific token counting if available.
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    async def check_rate_limit(self, model: str, text: Optional[str] = None):
        """Check rate limits before making an API call.
        
        Args:
            model: The model name/id
            text: Optional text to check token limits for
        """
        limiter = self.get_rate_limiter(model)
        tokens = self.estimate_tokens(text) if text else None
        await limiter.acquire(tokens)
        
        # Log current usage
        usage = limiter.get_current_usage()
        logger.debug(f"Rate limit usage for {model}: {usage}")
        
    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt."""
        pass
        
    @abstractmethod
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts."""
        pass
