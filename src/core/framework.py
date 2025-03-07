# Facade Pattern for a simplified interface that hides underlying complexity of the LLM framework
# Example: A single entry point for all LLM operations
#   framework = LLMFramework(api_keys)
#   result = await framework.process_file(file_path, model_config, templates, batch_mode=True)

# src/core/framework.py

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .factory import ClientFactory, ProcessorFactory
from ..models.config import ModelConfig

class LLMFramework:
    """Main facade for the LLM framework providing a simplified interface."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the framework with API keys for various providers.
        
        Args:
            api_keys: Dictionary mapping service names to API keys
        """
        self.client_factory = ClientFactory(api_keys)
        self.processor_factory = ProcessorFactory(self.client_factory)
    
    async def process_file(
        self,
        file_path: Path,
        model_config: ModelConfig,
        templates: List[Dict],
        batch_mode: bool = False,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> Path:
        """Process a file with multiple templates and models.
        
        Args:
            file_path: Path to the input file (TSV format)
            model_config: Configuration for the model
            templates: List of templates to apply
            batch_mode: Whether to use batch processing (when available)
            batch_size: Number of items to process in each batch
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Path: Path to the processed output file
        """
        processor = self.processor_factory.get_processor(
            batch_mode=batch_mode, 
            model_config=model_config
        )
        
        return await processor.process_file(
            file_path=file_path,
            templates=templates,
            model_config=model_config,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
    
    async def process_text(
        self,
        text: str,
        model_config: ModelConfig,
        template: Dict[str, str],
        batch_mode: bool = False
    ) -> str:
        """Process a single text input.
        
        Args:
            text: The input text to process
            model_config: Configuration for the model
            template: Template to apply
            batch_mode: Whether to use batch processing
            
        Returns:
            str: The processed response
        """
        processor = self.processor_factory.get_processor(
            batch_mode=batch_mode,
            model_config=model_config
        )
        
        return await processor.process_text(
            text=text,
            template=template,
            model_config=model_config
        )
    
    async def process_batch(
        self,
        items: List[str],
        model_config: ModelConfig,
        template: Dict[str, str],
        max_batch_size: int = 5000
    ) -> List[str]:
        """Process a batch of items.
        
        Args:
            items: List of text items to process
            model_config: Configuration for the model
            template: Template to apply
            max_batch_size: Maximum batch size
            
        Returns:
            List[str]: The processed responses
        """
        processor = self.processor_factory.get_processor(
            batch_mode=True,
            model_config=model_config
        )
        
        return await processor.process_batch(
            items=items,
            template=template,
            model_config=model_config,
            max_batch_size=max_batch_size
        )
