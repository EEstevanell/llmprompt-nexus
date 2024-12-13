import asyncio

import aiohttp
from src.rate_limiter import RateLimiter

class PerplexityClient:
    # Rate limits per model
    RATE_LIMITS = {
        "llama-3.1-sonar-small-128k-online": {"calls": 50, "period": 60},
        "llama-3.1-sonar-large-128k-online": {"calls": 50, "period": 60},
        "llama-3.1-sonar-huge-128k-online": {"calls": 50, "period": 60}
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.limiters = {
            model: RateLimiter(limits["calls"], limits["period"])
            for model, limits in self.RATE_LIMITS.items()
        }
    
    async def make_request(self, session, model: str, data: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Ensure model has a rate limiter
        if model not in self.limiters:
            raise ValueError(f"Unknown model: {model}")
        
        # Wait for rate limit
        await self.limiters[model].acquire()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get("Retry-After", retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))
                
        raise Exception("Max retries exceeded")
