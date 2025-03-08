import os
import json
import asyncio
import httpx
from typing import Dict, List, Any, Optional

from src.clients.base import BaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIClient(BaseClient):
    """Client for OpenAI's Chat API with parallel processing support."""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt."""
        try:
            # Prepare the messages
            messages = [{"role": "user", "content": prompt}]
            
            # Merge parameters
            request_params = {
                "model": model,
                "messages": messages
            }
            request_params.update(kwargs)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    json=request_params,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "error": error_msg,
                        "model": model
                    }
                
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                return {
                    "response": response_text,
                    "model": model,
                    "usage": result.get("usage", {})
                }
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "model": model
            }
