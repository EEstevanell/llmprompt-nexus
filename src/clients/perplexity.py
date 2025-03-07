# src/clients/perplexity.py
from clients.base import BaseClient
from rate_limiting.limiter import RateLimiter
import aiohttp
import asyncio
from typing import Dict, Any

class PerplexityClient(BaseClient):
    API_URL = "https://api.perplexity.ai/chat/completions"
    RATE_LIMITS = {
        "llama-3.1-sonar-small-128k-online": {"calls": 50, "period": 60},
        "llama-3.1-sonar-large-128k-online": {"calls": 50, "period": 60},
        "llama-3.1-sonar-huge-128k-online": {"calls": 50, "period": 60},
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
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with session.post(self.API_URL, headers=headers, json=data) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))
