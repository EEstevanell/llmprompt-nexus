import os
import json
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union

from llmprompt_nexus.clients.base import BaseClient
from llmprompt_nexus.utils.logger import get_logger

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
    
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], model: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for a single prompt.
        
        Args:
            prompt: Either a string prompt or a list of message dictionaries
            model: The model to use for generation
            **kwargs: Additional parameters to pass to the API
        """
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
