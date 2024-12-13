from datetime import datetime, timedelta
from collections import deque
import asyncio

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls  # Number of calls allowed
        self.period = period  # Time period in seconds
        self.calls = deque()  # Queue to track timestamps of calls
    
    async def acquire(self):
        now = datetime.now()
        
        # Remove timestamps older than our period
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            # Wait until oldest call expires
            sleep_time = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)
