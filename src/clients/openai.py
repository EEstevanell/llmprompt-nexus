# src/clients/openai.py
from clients.base import BaseClient
from rate_limiting.limiter import RateLimiter
import aiohttp
from typing import Dict, Any

class OpenAIClient(BaseClient):
    API_URL = "https://api.openai.com/v1/chat/completions"
    RATE_LIMITS = {
        "gpt-4o": {"calls": 500, "period": 60},
        "gpt-4o-mini": {"calls": 500, "period": 60},
        "gpt-3.5-turbo": {"calls": 500, "period": 60}
    }
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.limiters = {
            model: RateLimiter(limits["calls"], limits["period"])
            for model, limits in self.RATE_LIMITS.items()
        }
    
    def get_rate_limiter(self, model: str) -> RateLimiter:
        return self.limiters[model]
    
    async def make_request(self, session: aiohttp.ClientSession, model: str, data: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        await self.limiters[model].acquire()
        
        async with session.post(self.API_URL, headers=headers, json=data) as response:
            response.raise_for_status()
            return await response.json()
