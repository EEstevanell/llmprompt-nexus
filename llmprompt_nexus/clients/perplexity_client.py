import os
import json
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union

from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class PerplexityClient(BaseClient):
    """Client for Perplexity AI API with enhanced rate limiting."""
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._request_lock = asyncio.Lock()
    
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response with enhanced rate limiting."""
        try:
            # Handle different prompt types
            if isinstance(prompt, str):
                messages = []
                # Add system message if provided
                if 'system_message' in kwargs:
                    messages.append({"role": "system", "content": kwargs.pop('system_message')})
                messages.append({"role": "user", "content": prompt})
            else:
                messages = prompt
            
            # Merge parameters
            request_params = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            request_params.update(kwargs)
            
            rate_limiter = self.get_rate_limiter(model)
            
            # Use a lock to prevent race conditions in rate limiting
            async with self._request_lock:
                # Wait for rate limit clearance
                await rate_limiter.acquire()
                
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            self.API_URL,
                            headers=self.headers,
                            json=request_params,
                            timeout=30.0
                        )
                        
                        if response.status_code == 429:
                            rate_limiter.record_failure(is_429=True)
                            retry_after = int(response.headers.get('Retry-After', 60))
                            raise Exception(f"Rate limit exceeded. Retry after {retry_after}s")
                            
                        response.raise_for_status()
                        return response.json()
                        
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        rate_limiter.record_failure(is_429=True)
                    else:
                        rate_limiter.record_failure()
                    raise
                    
        except Exception as e:
            logger.error(f"Perplexity API error: {e}", exc_info=True)
            raise
            
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple prompts with automatic batching and rate limiting.
        
        This implementation uses the BaseClient's default batch processing logic
        which now includes proper rate limiting and retries.
        """
        return await super().generate_batch(prompts, model, **kwargs)
