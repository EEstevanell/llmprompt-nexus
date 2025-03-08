from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    id: str
    name: str
    provider: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, int]] = None
    
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
            
        # Validate supported providers
        supported_providers = ["openai", "perplexity"]
        if self.provider not in supported_providers:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Validate rate limits if present
        if self.rate_limits:
            rpm = self.rate_limits.get('rpm')
            if rpm is not None and (not isinstance(rpm, int) or rpm <= 0):
                raise ValueError(f"Invalid rpm rate limit: {rpm}")
        
        # Validate parameters if present
        if self.parameters and not isinstance(self.parameters, dict):
            raise ValueError(f"Parameters must be a dictionary")
    
    def get_rpm(self) -> int:
        """Get requests per minute limit with fallback to default."""
        if self.rate_limits and 'rpm' in self.rate_limits:
            return self.rate_limits['rpm']
        return 10  # Default RPM if not specified
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "parameters": self.parameters,
            "rate_limits": self.rate_limits
        }
