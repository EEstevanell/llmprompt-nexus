# src/models/registry.py
from typing import Dict
from models.config import ModelConfig

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
    
    def register(self, model_id: str, config: ModelConfig):
        self.models[model_id] = config
    
    def get_model(self, model_id: str) -> ModelConfig:
        return self.models[model_id]
    
    def list_models(self) -> Dict[str, ModelConfig]:
        return self.models.copy()

# Create global registry
registry = ModelRegistry()

# Register default models
registry.register("sonar-small", ModelConfig(
    name="llama-3.1-sonar-small-128k-online",
    api="perplexity",
    parameters={"temperature": 0.2}
))

# Register default models
registry.register("sonar-huge", ModelConfig(
    name="llama-3.1-sonar-huge-128k-online",
    api="perplexity",
    parameters={"temperature": 0.2}
))

registry.register("gpt-4o", ModelConfig(
    name="gpt-4o",
    api="openai",
    parameters={"temperature": 0.7}
))

registry.register("gpt-4o-mini", ModelConfig(
    name="gpt-4o-mini",
    api="openai",
    parameters={"temperature": 0.7}
))