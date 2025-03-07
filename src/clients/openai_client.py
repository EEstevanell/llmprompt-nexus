import httpx
import asyncio
from typing import Dict, List, Any, Optional
import tiktoken

from src.clients.base import BaseClient
from src.models.registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIClient(BaseClient):
    """Client for interacting with OpenAI API."""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Default to GPT-4 encoding
        logger.info("Initialized OpenAI client")
        
    def estimate_tokens(self, text: str) -> int:
        """Get accurate token count using tiktoken."""
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
            
            # Check rate limits with accurate token count
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
                            "model": actual_model_name,  # Use the actual model name from config
                            "messages": messages,
                            **request_params
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                        raise ValueError(f"OpenAI API error: {response.status_code} - {response.text}")
                        
                    result = response.json()
                    logger.debug(f"Generated response with {self.estimate_tokens(result['choices'][0]['message']['content'])} tokens")
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "model": model,
                        "usage": result.get("usage", {})
                    }
                except httpx.TimeoutException:
                    logger.error(f"Request to OpenAI API timed out for model {model}")
                    raise ValueError(f"Request timed out for model {model}")
                except Exception as e:
                    logger.error(f"Error calling OpenAI API: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def generate_batch(self, prompts: List[str], model: str, batch_size: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            model: The model identifier
            batch_size: Number of concurrent requests (default: 5)
            **kwargs: Additional model parameters
            
        Returns:
            List of response dictionaries
        """
        try:
            # Calculate total tokens for the batch
            total_tokens = sum(self.estimate_tokens(prompt) for prompt in prompts)
            await self.check_rate_limit(model, total_tokens)
            
            logger.debug(f"Generating batch responses for {len(prompts)} prompts with total {total_tokens} tokens")
            
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
            
            total_response_tokens = sum(self.estimate_tokens(r['response']) for r in results if 'response' in r)
            logger.debug(f"Generated {len(results)} responses with total {total_response_tokens} tokens")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch responses: {str(e)}")
            raise
