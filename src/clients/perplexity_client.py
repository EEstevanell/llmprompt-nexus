import httpx
import asyncio
from typing import Dict, List, Any, Optional
import tiktoken  # Perplexity uses similar tokenization to GPT models

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
        # Use GPT tokenizer since Perplexity uses similar tokenization
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        logger.info("Initialized Perplexity client")
        
    def estimate_tokens(self, text: str) -> int:
        """Get approximate token count using tiktoken."""
        return len(self.encoding.encode(text))
        
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt.
        
        Args:
            prompt: The prompt to send to the model
            model: The model identifier (as defined in config)
            **kwargs: Additional model parameters
            
        Returns:
            Response dictionary with results
            
        Raises:
            ValueError: If model doesn't exist or API call fails
        """
        try:
            # Get model config and validate it exists
            model_config = registry.get_model(model)
            actual_model_name = model_config.name
            
            # Check rate limits with token count
            await self.check_rate_limit(model, prompt)
            
            logger.debug(f"Generating response for prompt with {self.estimate_tokens(prompt)} tokens")
            
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
                            "model": actual_model_name,
                            "messages": messages,
                            **request_params
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                        raise ValueError(f"Perplexity API error: {response.status_code} - {response.text}")
                        
                    result = response.json()
                    logger.debug(f"Generated response with {self.estimate_tokens(result['choices'][0]['message']['content'])} tokens")
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
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts sequentially.
        
        Perplexity API doesn't support batch operations, so we process one at a time.
        
        Args:
            prompts: List of prompts to process
            model: The model identifier
            **kwargs: Additional model parameters
            
        Returns:
            List of response dictionaries
        """
        logger.warning("Batch processing not supported by Perplexity API, processing sequentially")
        try:
            results = []
            for prompt in prompts:
                try:
                    result = await self.generate(prompt, model, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing prompt in batch: {e}")
                    results.append({"error": str(e), "model": model})
            return results
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
