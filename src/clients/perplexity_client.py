import httpx
import asyncio
from typing import Dict, List, Any

from src.clients.base import BaseClient
from src.models.registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityClient(BaseClient):
    """Client for interacting with Perplexity API."""
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt.
        
        Args:
            prompt: The prompt to send to the model
            model: Model identifier (as defined in config)
            **kwargs: Additional model parameters
            
        Returns:
            Response dictionary with results
            
        Raises:
            ValueError: If model doesn't exist or API call fails
        """
        # Get model config and validate it exists
        try:
            model_config = registry.get_model(model)
            actual_model_name = model_config.name
        except ValueError as e:
            logger.error(f"Invalid model requested: {model}")
            raise e
        
        # Merge default parameters from config with user-provided parameters
        request_params = {}
        if model_config.parameters:
            request_params.update(model_config.parameters)
        if kwargs:
            request_params.update(kwargs)
        
        # Construct messages from prompt
        messages = [{"role": "user", "content": prompt}]
        
        # Respect rate limits using the limiter from BaseClient
        rate_limiter = self.get_rate_limiter(model)
        await rate_limiter.acquire()
        
        logger.info(f"Generating with {model} (API model: {actual_model_name})")
        
        # Make API request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    json={
                        "model": actual_model_name,  # Use the actual model name from config
                        "messages": messages,
                        **request_params
                    },
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                    raise ValueError(f"Perplexity API error: {response.status_code} - {response.text}")
                    
                result = response.json()
                return {
                    "response": result["choices"][0]["message"]["content"],
                    "model": model,
                    "usage": result.get("usage", {})
                }
            except httpx.TimeoutException:
                logger.error(f"Request to Perplexity API timed out for model {model}")
                raise ValueError(f"Request timed out for model {model}")
            except Exception as e:
                logger.error(f"Error calling Perplexity API: {e}")
                raise
            
    async def generate_batch(self, prompts: List[str], model: str, batch_size: int = 2, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            model: The model identifier
            batch_size: Number of concurrent requests (default: 2, Perplexity has stricter rate limits)
            **kwargs: Additional model parameters
            
        Returns:
            List of response dictionaries
        """
        # Process prompts in batches with limited concurrency
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tasks = [self.generate(prompt, model, **kwargs) for prompt in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                    results.append({"error": str(result), "model": model})
                else:
                    results.append(result)
        
        return results
