"""Rate limiting implementation with advanced features."""
from datetime import datetime, timedelta
from collections import deque
import asyncio
from typing import Dict, Any, Optional
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """Rate limiter with exponential backoff and failure tracking."""
    
    def __init__(self, max_calls: int, period: int):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()  # Successful calls
        self.failed_calls = deque()  # Failed calls
        self.last_429 = None  # Last time we got a 429
        self.backoff_until = None  # Time until we should maintain reduced rate
        self._lock = asyncio.Lock()  # Lock for thread safety
        logger.debug(f"Initialized rate limiter: {max_calls} calls per {period}s")
    
    def _clean_old_calls(self, now: datetime) -> None:
        """Clean up calls older than our period."""
        period_delta = timedelta(seconds=self.period)
        while self.calls and now - self.calls[0] > period_delta:
            self.calls.popleft()
        while self.failed_calls and now - self.failed_calls[0] > period_delta:
            self.failed_calls.popleft()
    
    def _should_backoff(self, now: datetime) -> tuple[bool, float]:
        """Determine if we should apply backoff and how much."""
        if not self.backoff_until or now > self.backoff_until:
            return False, 0
            
        # Calculate remaining backoff time
        remaining = (self.backoff_until - now).total_seconds()
        return True, remaining
    
    def _calculate_delay(self, current_calls: int) -> float:
        """Calculate delay needed to stay within rate limits."""
        if not self.calls:
            return 0
            
        # If we're at max calls, calculate delay until oldest call expires
        if current_calls >= self.max_calls:
            now = datetime.now()
            delay = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
            return max(0, delay)
        
        return 0
    
    async def acquire(self) -> None:
        """Acquire permission to make an API call."""
        async with self._lock:
            now = datetime.now()
            
            # Clean up old calls
            self._clean_old_calls(now)
            
            # Check if we need to backoff
            should_backoff, backoff_time = self._should_backoff(now)
            if should_backoff:
                delay = min(backoff_time, 5)  # Cap individual delays at 5s
                logger.warning(f"In backoff period, adding {delay:.2f}s delay")
                await asyncio.sleep(delay)
            
            # Count current calls including failed ones
            current_calls = len(self.calls) + len(self.failed_calls)
            
            # Calculate required delay
            delay = self._calculate_delay(current_calls)
            
            if delay > 0:
                logger.warning(f"Rate limit reached, waiting {delay:.2f}s")
                await asyncio.sleep(delay)
            
            # Record the call
            self.calls.append(datetime.now())
            
            # Reduce max_calls temporarily if we've seen failures
            if len(self.failed_calls) > 0:
                effective_limit = max(1, int(self.max_calls * 0.8))  # Reduce by 20%
                logger.debug(f"Reducing effective rate limit to {effective_limit} due to recent failures")
    
    def record_failure(self, is_429: bool = False) -> None:
        """Record a failed API call."""
        now = datetime.now()
        self.failed_calls.append(now)
        
        if is_429:
            self.last_429 = now
            # Extend or set backoff period
            if not self.backoff_until or now > self.backoff_until:
                self.backoff_until = now + timedelta(seconds=60)  # Start with 1 minute
            else:
                # Double the remaining backoff time
                remaining = (self.backoff_until - now).total_seconds()
                self.backoff_until = now + timedelta(seconds=min(remaining * 2, 300))  # Cap at 5 minutes
        
        # Remove the successful call record since it failed
        if self.calls:
            self.calls.pop()
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        now = datetime.now()
        self._clean_old_calls(now)
        
        stats = {
            "total_calls": len(self.calls),
            "failed_calls": len(self.failed_calls),
            "max_calls": self.max_calls,
            "remaining": max(0, self.max_calls - len(self.calls) - len(self.failed_calls)),
            "last_429_ago": None,
            "backoff_remaining": None
        }
        
        if self.last_429:
            stats["last_429_ago"] = (now - self.last_429).total_seconds()
            
        if self.backoff_until and now < self.backoff_until:
            stats["backoff_remaining"] = (self.backoff_until - now).total_seconds()
            
        return stats
