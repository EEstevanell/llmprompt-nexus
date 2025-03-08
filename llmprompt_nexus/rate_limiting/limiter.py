# src/rate_limiting/limiter.py
from datetime import datetime, timedelta
from collections import deque
import asyncio
from typing import Dict, Any
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        logger.info(f"Initialized rate limiter: {max_calls} calls per {period}s")
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        now = datetime.now()
        
        # Clean up old calls
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
        
        # Check call rate limits
        if len(self.calls) >= self.max_calls:
            sleep_time = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, waiting {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Record the call
        self.calls.append(now)
        logger.debug(f"Acquired rate limit: {len(self.calls)}/{self.max_calls} calls")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        now = datetime.now()
        # Clean up expired entries first
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
            
        return {
            "calls": len(self.calls),
            "max_calls": self.max_calls,
            "period": self.period
        }
