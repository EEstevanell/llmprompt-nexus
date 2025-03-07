from typing import Dict, List, Any

from clients.base import BaseClient
from src.models.model_config import ModelConfig
from src.processors.base_processor import BaseProcessor

class PerplexityProcessor(BaseProcessor):
    """Processor for Perplexity models."""
    
    def __init__(self, client: BaseClient, model_config: ModelConfig, templates: Dict):
        self.client = client
        self.model_config = model_config
        self.templates = templates
        
    async def process_item(self, item: Dict) -> Dict[str, Any]:
        """Process a single item."""
        prompt = self._format_prompt(item)
        return await self.client.generate(
            prompt=prompt,
            model=self.model_config.model_name,
            temperature=self.model_config.temperature
        )
        
    async def process_batch(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple items as a batch."""
        prompts = [self._format_prompt(item) for item in items]
        return await self.client.generate_batch(
            prompts=prompts,
            model=self.model_config.model_name,
            temperature=self.model_config.temperature
        )
        
    def _format_prompt(self, item: Dict) -> str:
        """Format the prompt using the template and item data."""
        template = self.templates.get(item.get("intention", "default"), "")
        # Replace placeholders in the template with item data
        prompt = template
        for key, value in item.items():
            if isinstance(value, str):
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, value)
        return prompt
