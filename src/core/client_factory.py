from typing import Dict, Optional, List, Set
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
    
    # Class-level registry of supported providers and their client classes
    SUPPORTED_PROVIDERS = {
        "openai": OpenAIClient,
        "perplexity": PerplexityClient
    }
    
    # Providers that support batch processing
    BATCH_SUPPORTED_PROVIDERS = {"openai"}
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the client factory.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys
        """
        self.api_keys = api_keys
        self.clients = {}
        self._validate_api_keys()
        
    def _validate_api_keys(self) -> None:
        """Validate that all required API keys are present.
        
        Raises:
            ValueError: If any required API key is missing
        """
        # Get available providers from registry
        available_providers = set(registry.list_providers())
        
        # Check if we have API keys for all available providers
        missing_keys = [provider for provider in available_providers 
                        if provider not in self.api_keys or not self.api_keys[provider]]
        
        if missing_keys:
            logger.warning(f"Missing API keys for providers: {', '.join(missing_keys)}")
        
        # Validate we have at least one valid API key
        if not any(key for key in self.api_keys.values() if key):
            raise ValueError("No valid API keys provided")
            
    def get_supported_providers(self) -> List[str]:
        """Get list of providers supported by this factory.
        
        Returns:
            List of provider names that this factory can create clients for
        """
        return list(self.SUPPORTED_PROVIDERS.keys())
        
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
        
        # Verify provider is supported
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Provider '{provider}' for model '{model_id}' is not supported. "
                            f"Supported providers: {', '.join(self.get_supported_providers())}")
        
        # For providers with batch mode, check if batch is requested and available
        if use_batch and provider in self.BATCH_SUPPORTED_PROVIDERS:
            batch_config = registry.get_batch_config(provider)
            if batch_config and batch_config.enabled:
                return self.get_client(provider, use_batch=True)
            else:
                # Batch requested but not available, log warning and fall back to standard client
                logger.warning(f"Batch processing requested for {provider} but not available or disabled")
        
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
        # Validate provider is supported
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. "
                            f"Available providers: {', '.join(self.get_supported_providers())}")
            
        # Check if batch is requested but not supported
        if use_batch and provider not in self.BATCH_SUPPORTED_PROVIDERS:
            logger.warning(f"Batch processing requested for {provider} but not supported, using standard client")
            use_batch = False
            
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
        try:
            if provider == "openai":
                if use_batch:
                    client = OpenAIBatchClient(api_key)
                    logger.info("Created OpenAI Batch client")
                else:
                    client = OpenAIClient(api_key)
                    logger.info("Created OpenAI client")
            elif provider == "perplexity":
                client = PerplexityClient(api_key)
                logger.info("Created Perplexity client")
            
            # Cache the client for future use
            self.clients[cache_key] = client
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for {provider}: {str(e)}")
            raise
