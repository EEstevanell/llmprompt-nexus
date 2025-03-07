# src/models/config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    name: str
    api: str
    parameters: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "api": self.api,
            **(self.parameters or {})
        }
