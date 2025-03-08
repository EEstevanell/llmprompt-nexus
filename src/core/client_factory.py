from typing import Dict
from src.clients.base import BaseClient
from src.clients.openai_client import OpenAIClient
from src.clients.perplexity_client import PerplexityClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_client(api_name: str, api_key: str) -> BaseClient:
    """Create an API client instance.
    
    Args:
        api_name: Name of the API provider (openai, perplexity)
        api_key: API key for authentication
        
    Returns:
        An instance of the appropriate client class
        
    Raises:
        ValueError: If the API provider is not supported
    """
    clients = {
        "openai": OpenAIClient,
        "perplexity": PerplexityClient
    }
    
    if api_name not in clients:
        raise ValueError(f"Unsupported API provider: {api_name}")
    
    client_class = clients[api_name]
    return client_class(api_key)
