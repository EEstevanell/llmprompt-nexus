# src/models/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RateLimit:
    """Configuration for rate limits of a model."""
    rpm: Optional[int] = None  # Requests per minute
    period: Optional[int] = None  # Period in seconds

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: str
    max_tokens: int
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, int]] = None  # Rate limits (rpm)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        if not self.name:
            raise ValueError("Model name is required")
        if not self.provider:
            raise ValueError("Model provider is required")
        if self.max_tokens <= 0:
            raise ValueError(f"Max tokens must be greater than 0, got {self.max_tokens}")
            
        # Validate supported providers (can be extended as needed)
        supported_providers = ["openai", "perplexity"]
        if self.provider not in supported_providers:
            logger.warning(f"Provider '{self.provider}' may not be fully supported. "
                          f"Supported providers: {', '.join(supported_providers)}")
        
        # Validate rate limits if present
        if self.rate_limits:
            for limit_type, value in self.rate_limits.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid rate limit for {limit_type}: {value}")
        
        # Validate parameters if present
        if self.parameters and not isinstance(self.parameters, dict):
            raise ValueError(f"Parameters must be a dictionary, got {type(self.parameters)}")
    
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
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate batch API configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        if self.max_requests_per_batch <= 0:
            raise ValueError(f"Max requests per batch must be positive, got {self.max_requests_per_batch}")
        if self.max_file_size_bytes <= 0:
            raise ValueError(f"Max file size must be positive, got {self.max_file_size_bytes}")
        
        # Validate rate limits if present
        if self.rate_limits:
            for operation, limits in self.rate_limits.items():
                if not isinstance(limits, dict):
                    raise ValueError(f"Rate limit for {operation} must be a dictionary")
                for key, value in limits.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        raise ValueError(f"Invalid rate limit value for {operation}.{key}: {value}")
