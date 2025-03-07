# src/processors/base_processor.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..clients.base import BaseClient
from ..models.config import ModelConfig

class BaseProcessor(ABC):
    """Abstract base class for all processors."""
    
    def __init__(self, client: BaseClient):
        """Initialize the processor with a client.
        
        Args:
            client: The API client to use
        """
        self.client = client
    
    @abstractmethod
    async def process_file(
        self,
        file_path: Path,
        templates: List[Dict],
        model_config: ModelConfig,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> Path:
        """Process a file with multiple templates.
        
        Args:
            file_path: Path to the input file
            templates: List of templates to apply
            model_config: Configuration for the model
            batch_size: Number of items to process in each batch
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Path: Path to the processed output file
        """
        pass
    
    @abstractmethod
    async def process_text(
        self,
        text: str,
        template: Dict[str, str],
        model_config: ModelConfig
    ) -> str:
        """Process a single text input.
        
        Args:
            text: The input text to process
            template: Template to apply
            model_config: Configuration for the model
            
        Returns:
            str: The processed response
        """
        pass
    
    @abstractmethod
    async def process_batch(
        self,
        items: List[str],
        template: Dict[str, str],
        model_config: ModelConfig,
        max_batch_size: int = 5000
    ) -> List[str]:
        """Process a batch of items.
        
        Args:
            items: List of text items to process
            template: Template to apply
            model_config: Configuration for the model
            max_batch_size: Maximum batch size
            
        Returns:
            List[str]: The processed responses
        """
        pass
