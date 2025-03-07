#!/usr/bin/env python
# examples/test_model_config.py

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import our components
from src.models.registry import registry
from src.clients.openai_client import OpenAIClient
from src.clients.openai_batch import OpenAIBatchClient
from src.clients.perplexity_client import PerplexityClient

async def test_models():
    """Test loading and using model configurations from YAML."""
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment")
    
    if not perplexity_api_key:
        print("Warning: PERPLEXITY_API_KEY not found in environment")
    
    # 1. List all available models from registry
    print("Available models:")
    for model_id in registry.list_models():
        model_config = registry.get_model(model_id)
        print(f"  - {model_id} (Provider: {model_config.provider})")
    
    print("\nAvailable providers:", registry.list_providers())
    
    # 2. List models by provider
    print("\nOpenAI models:")
    for model_id in registry.list_models(provider="openai"):
        print(f"  - {model_id}")
    
    print("\nPerplexity models:")
    for model_id in registry.list_models(provider="perplexity"):
        print(f"  - {model_id}")
    
    # 3. Get batch configuration for OpenAI
    batch_config = registry.get_batch_config("openai")
    if batch_config:
        print("\nOpenAI Batch API Configuration:")
        print(f"  Enabled: {batch_config.enabled}")
        print(f"  Max requests per batch: {batch_config.max_requests_per_batch}")
        print(f"  Max file size: {batch_config.max_file_size_bytes / (1024 * 1024):.1f} MB")
        
        if batch_config.rate_limits:
            print("  Rate limits:")
            for operation, limits in batch_config.rate_limits.items():
                print(f"    - {operation}: {limits.get('calls', 'N/A')} calls per {limits.get('period', 'N/A')} seconds")
    
    # 4. Create clients and test model validation
    if openai_api_key:
        try:
            # Initialize OpenAI client with API key from environment
            print("\nTesting OpenAI client with model from config...")
            openai_client = OpenAIClient(openai_api_key)
            
            # Test with a valid model from config
            model_id = "gpt-3.5-turbo"  # This should be in our YAML config
            model_config = registry.get_model(model_id)
            
            print(f"Model {model_id} configuration:")
            print(f"  Name: {model_config.name}")
            print(f"  Provider: {model_config.provider}")
            print(f"  Description: {model_config.description}")
            print(f"  Max tokens: {model_config.max_tokens}")
            
            if model_config.rate_limits:
                print("  Rate limits:")
                for key, value in model_config.rate_limits.items():
                    print(f"    - {key}: {value}")
            
            if model_config.parameters:
                print("  Default parameters:")
                for key, value in model_config.parameters.items():
                    print(f"    - {key}: {value}")
            
            # Test batch client if batch API is enabled
            if batch_config and batch_config.enabled:
                print("\nTesting OpenAI batch client...")
                batch_client = OpenAIBatchClient(openai_api_key)
                print(f"  Max requests per batch: {batch_client.MAX_REQUESTS_PER_BATCH}")
                print(f"  Max file size: {batch_client.MAX_FILE_SIZE_BYTES / (1024 * 1024):.1f} MB")
        
        except ValueError as e:
            print(f"Error: {e}")
    
    # 5. Test with an invalid model to verify error handling
    try:
        print("\nTesting with invalid model...")
        invalid_model = "non-existent-model"
        model_config = registry.get_model(invalid_model)
    except ValueError as e:
        print(f"Expected error (good): {e}")

if __name__ == "__main__":
    asyncio.run(test_models())