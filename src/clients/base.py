# src/clients/base.py
from abc import ABC, abstractmethod
import aiohttp
from typing import Dict, Any

class BaseClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    async def make_request(self, session: aiohttp.ClientSession, model: str, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_rate_limiter(self, model: str):
        pass
