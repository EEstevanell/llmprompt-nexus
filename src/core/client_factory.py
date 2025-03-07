from typing import Dict, Optional
from src.clients.base import BaseClient
from src.clients.openai_client import OpenAIClient
from src.clients.perplexity_client import PerplexityClient
from src.clients.openai_batch import OpenAIBatchClient
from src.models.registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ClientFactory:
    """Factory for creating API clients based on model configurations.
    
    This factory creates and caches clients for different providers,
    using the model registry to determine the appropriate client type.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the client factory.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys
        """
        self.api_keys = api_keys
        self.clients = {}
        
    def get_client_for_model(self, model_id: str, use_batch: bool = False) -> BaseClient:
        """Get the appropriate client for a specific model.
        
        Args:
            model_id: The model identifier from the registry
            use_batch: Whether to use batch client if available
            
        Returns:
            The appropriate client instance
            
        Raises:
            ValueError: If model doesn't exist or API key is missing
        """
        # Get model configuration from registry to determine provider
        model_config = registry.get_model(model_id)
        provider = model_config.provider
        
        # For OpenAI with batch mode, use the batch client
        if provider == "openai" and use_batch:
            batch_config = registry.get_batch_config("openai")
            if batch_config and batch_config.enabled:
                return self.get_client(provider, use_batch=True)
        
        # Get standard client for the provider
        return self.get_client(provider)
    
    def get_client(self, provider: str, use_batch: bool = False) -> BaseClient:
        """Get the appropriate client for the provider type.
        
        Args:
            provider: The provider name (e.g., "openai", "perplexity")
            use_batch: Whether to use batch processing client if available
            
        Returns:
            The client instance
            
        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        # Create a cache key that includes batch preference
        cache_key = f"{provider}{'_batch' if use_batch else ''}"
        
        # Return cached client if available
        if cache_key in self.clients:
            return self.clients[cache_key]
        
        # Check if API key is available
        api_key = self.api_keys.get(provider)
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        # Create the appropriate client instance
        if provider == "openai":
            if use_batch:
                client = OpenAIBatchClient(api_key)
                logger.info(f"Created OpenAI Batch client")
            else:
                client = OpenAIClient(api_key)
                logger.info(f"Created OpenAI client")
        elif provider == "perplexity":
            client = PerplexityClient(api_key)
            logger.info(f"Created Perplexity client")
        else:
            available_providers = registry.list_providers()
            raise ValueError(f"Unsupported provider: {provider}. Available providers: {available_providers}")
        
        # Cache the client for future use
        self.clients[cache_key] = client
        return client
