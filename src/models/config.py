# src/models/config.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RateLimit:
    """Configuration for rate limits of a model."""
    rpm: Optional[int] = None  # Requests per minute
    tpm: Optional[int] = None  # Tokens per minute
    calls: Optional[int] = None  # Specific calls (for batch operations)
    period: Optional[int] = None  # Period in seconds

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: str  # e.g., "openai", "perplexity"
    description: str = ""
    max_tokens: int = 4096
    rate_limits: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "max_tokens": self.max_tokens,
            "rate_limits": self.rate_limits,
            "parameters": self.parameters
        }
    
    @property
    def id(self) -> str:
        """Get model ID (for backward compatibility)."""
        return self.name
    
    @property
    def api(self) -> str:
        """Get API provider (for backward compatibility)."""
        return self.provider
    
    @property
    def model_name(self) -> str:
        """Get model name (for backward compatibility)."""
        return self.name
    
    @property
    def temperature(self) -> float:
        """Get temperature (for backward compatibility)."""
        return self.parameters.get("temperature", 0.7) if self.parameters else 0.7

@dataclass
class BatchAPIConfig:
    """Configuration for batch API operations."""
    enabled: bool = False
    max_requests_per_batch: int = 50000
    max_file_size_bytes: int = 209715200  # 200 MB
    rate_limits: Optional[Dict[str, Dict[str, int]]] = None
