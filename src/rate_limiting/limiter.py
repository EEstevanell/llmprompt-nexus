# src/rate_limiting/limiter.py
from datetime import datetime, timedelta
from collections import deque
import asyncio

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    async def acquire(self):
        now = datetime.now()
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            sleep_time = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)
