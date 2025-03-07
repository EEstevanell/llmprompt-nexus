# src/rate_limiting/limiter.py
from datetime import datetime, timedelta
from collections import deque
import asyncio
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    def __init__(self, max_calls: int, period: int, max_tokens_per_min: Optional[int] = None):
        self.max_calls = max_calls
        self.period = period
        self.max_tokens_per_min = max_tokens_per_min
        self.calls = deque()
        self.tokens = deque()  # Track token usage with timestamps
        logger.info(f"Initialized rate limiter: {max_calls} calls per {period}s" + 
                   (f", {max_tokens_per_min} tokens per minute" if max_tokens_per_min else ""))
    
    async def acquire(self, tokens: Optional[int] = None):
        """Acquire permission to make an API call."""
        now = datetime.now()
        
        # Clean up old calls
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
            
        # Clean up old token counts
        if self.max_tokens_per_min:
            while self.tokens and now - self.tokens[0]["timestamp"] > timedelta(minutes=1):
                self.tokens.popleft()
        
        # Check call rate limits
        if len(self.calls) >= self.max_calls:
            sleep_time = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, waiting {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Check token rate limits if applicable
        if tokens and self.max_tokens_per_min:
            current_tokens = sum(t["count"] for t in self.tokens)
            if current_tokens + tokens > self.max_tokens_per_min:
                # Wait until enough tokens are available
                oldest_tokens = self.tokens[0] if self.tokens else {"timestamp": now - timedelta(minutes=1)}
                sleep_time = (oldest_tokens["timestamp"] + timedelta(minutes=1) - now).total_seconds()
                if sleep_time > 0:
                    logger.warning(f"Token limit reached, waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Clean up expired tokens after waiting
                    while self.tokens and now - self.tokens[0]["timestamp"] > timedelta(minutes=1):
                        self.tokens.popleft()
        
        # Record the call and tokens
        self.calls.append(now)
        if tokens and self.max_tokens_per_min:
            self.tokens.append({"timestamp": now, "count": tokens})
            
        logger.debug(f"Acquired rate limit: {len(self.calls)}/{self.max_calls} calls" +
                    (f", {sum(t['count'] for t in self.tokens)}/{self.max_tokens_per_min} tokens/min" 
                     if self.max_tokens_per_min else ""))
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        now = datetime.now()
        # Clean up expired entries first
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
        while self.tokens and now - self.tokens[0]["timestamp"] > timedelta(minutes=1):
            self.tokens.popleft()
            
        return {
            "calls": len(self.calls),
            "max_calls": self.max_calls,
            "period": self.period,
            "tokens_per_min": sum(t["count"] for t in self.tokens) if self.tokens else 0,
            "max_tokens_per_min": self.max_tokens_per_min
        }
