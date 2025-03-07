# src/core/factory.py

from typing import Dict, Any, Optional, Type, Union

from ..clients.base import BaseClient
from ..clients.openai import OpenAIClient
from ..clients.openai.batch_client import OpenAIBatchClient
from ..clients.perplexity import PerplexityClient
from ..processors.base_processor import BaseProcessor
from ..processors.async_processor import AsyncProcessor
from ..processors.batch_processor import BatchProcessor
from ..models.config import ModelConfig

class ClientFactory:
    """Factory for creating API clients."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the factory with API keys.
        
        Args:
            api_keys: Dictionary mapping service names to API keys
        """
        self.api_keys = api_keys
        self.clients = {}
    
    def get_client(self, api_name: str, batch_mode: bool = False) -> BaseClient:
        """Get a client for the specified API.
        
        Args:
            api_name: Name of the API service ("openai" or "perplexity")
            batch_mode: Whether to use batch processing if available
            
        Returns:
            BaseClient: The appropriate API client
            
        Raises:
            ValueError: If the API name is not supported
        """
        key = f"{api_name}_{batch_mode}"
        
        if key not in self.clients:
            if api_name == "openai":
                if batch_mode:
                    self.clients[key] = OpenAIBatchClient(self.api_keys["openai"])
                else:
                    self.clients[key] = OpenAIClient(self.api_keys["openai"])
            elif api_name == "perplexity":
                self.clients[key] = PerplexityClient(self.api_keys["perplexity"])
            else:
                raise ValueError(f"Unsupported API: {api_name}")
        
        return self.clients[key]

class ProcessorFactory:
    """Factory for creating processors."""
    
    def __init__(self, client_factory: ClientFactory):
        """Initialize the factory with a client factory.
        
        Args:
            client_factory: Factory for creating API clients
        """
        self.client_factory = client_factory
        self.processors = {}
    
    def get_processor(
        self, 
        batch_mode: bool, 
        model_config: ModelConfig
    ) -> BaseProcessor:
        """Get a processor based on the specified strategy.
        
        Args:
            batch_mode: Whether to use batch processing
            model_config: Configuration for the model
            
        Returns:
            BaseProcessor: The appropriate processor
        """
        api_name = model_config.api
        
        # Check if batch mode is supported for this API
        supports_batch = (api_name == "openai")
        use_batch = batch_mode and supports_batch
        
        key = f"{api_name}_{use_batch}"
        
        if key not in self.processors:
            client = self.client_factory.get_client(api_name, use_batch)
            
            if use_batch:
                self.processors[key] = BatchProcessor(client)
            else:
                self.processors[key] = AsyncProcessor(client)
        
        return self.processors[key]
