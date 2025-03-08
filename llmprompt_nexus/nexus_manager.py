"""
NexusManager - A unified interface for interacting with various LLM providers.

This module provides a high-level framework for working with different Language Model APIs
through a consistent interface, with support for:
- Template-based interactions
- Rate limiting
- Batch processing
- Multiple providers (OpenAI, Perplexity, etc.)
"""

import asyncio
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any, Union

from llmprompt_nexus.clients.client_factory import create_client
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.processors.factory import create_processor
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.models.registry import registry as model_registry
from llmprompt_nexus.templates import load_template
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class NexusManager:
    """
    A unified framework for interacting with various LLM providers through a consistent interface.
    
    This framework provides:
    - Multi-provider support through a unified interface
    - Template-based interactions for different use cases
    - Built-in rate limiting and API key management
    - Batch processing capabilities
    - Asynchronous processing
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the framework with API keys for different providers."""
        self.api_keys = api_keys
    
    def get_client(self, api_name: str):
        """Get or create an API client instance."""
        if api_name not in self.api_keys:
            raise ValueError(f"No API key provided for {api_name}")
        return create_client(api_name, self.api_keys[api_name])
    
    async def generate(self, prompt: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using a specific model."""
        model_config = model_registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        processor = create_processor(client, model_config)
        
        return await processor.process_item({
            'prompt': prompt,
            'model': model_config.name,
            **kwargs
        })
    
    async def generate_batch(self, prompts: List[str], model_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts in parallel."""
        model_config = model_registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        processor = create_processor(client, model_config)
        
        items = [{'prompt': prompt, 'model': model_config.name, **kwargs} for prompt in prompts]
        return await processor.process_batch(items)
    
    async def run_with_model(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        model_id: str,
        template_name: Optional[str] = None,
        template_config: Optional[Dict[str, Any]] = None,
        batch_mode: bool = False,
        global_variables: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process input data with a specific model using an optional template.
        
        Args:
            input_data: Single dictionary or list of dictionaries containing input data
            model_id: The model ID to use for processing
            template_name: Optional name of template to load
            template_config: Optional template configuration dictionary
            batch_mode: Whether to process as a batch (if input_data is a list)
            global_variables: Optional dictionary of variables that apply to all examples
        """
        model_config = model_registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        
        template = None
        if template_name or template_config:
            template = load_template(
                template_name or "custom_template",
                template_config
            )
        
        processor = create_processor(client, model_config, template)
        
        # Handle global variables if provided
        if global_variables:
            if batch_mode and isinstance(input_data, list):
                input_data = [{**global_variables, **item} for item in input_data]
            elif not batch_mode and isinstance(input_data, dict):
                input_data = {**global_variables, **input_data}
        
        try:
            if batch_mode and isinstance(input_data, list):
                return await processor.process_batch(input_data)
            elif not batch_mode and isinstance(input_data, dict):
                return await processor.process_item(input_data)
            else:
                raise ValueError("Input data type must match batch_mode setting")
        except Exception as e:
            logger.error(f"Error processing with model {model_id}: {str(e)}")
            raise
    
    async def process_file(
        self,
        file_path: Path,
        model_config: ModelConfig,
        template_name: Optional[str] = None,
        template_config: Optional[Dict[str, Any]] = None,
        batch_mode: bool = False,
        batch_size: int = 10
    ) -> Path:
        """
        Process a TSV file with the specified model configuration.
        
        Args:
            file_path: Path to the input TSV file
            model_config: Configuration for the model to use
            template_name: Optional name of template to load
            template_config: Optional template configuration dictionary
            batch_mode: Whether to use batch processing
            batch_size: Size of batches when batch_mode is True
        
        Returns:
            Path to the output TSV file with results
        """
        logger.info(f"Processing file {file_path} with model {model_config.id}")
        
        template = None
        if template_name or template_config:
            template = load_template(
                template_name or "custom_template",
                template_config
            )

        try:
            df = pd.read_csv(file_path, sep='\t')
            if df.empty:
                raise ValueError(f"Input file {file_path} is empty")
            
            logger.info(f"Read {len(df)} rows from {file_path}")
            
            client = self.get_client(model_config.provider)
            processor = create_processor(client, model_config, template)
            
            if batch_mode:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size].to_dict('records')
                    results = await processor.process_batch(batch)
                    
                    for j, result in enumerate(results):
                        idx = i + j
                        if idx < len(df):
                            df.at[idx, 'response'] = result.get('response', '')
                            df.at[idx, 'model'] = model_config.id
            else:
                for idx, row in df.iterrows():
                    result = await processor.process_item(row.to_dict())
                    df.at[idx, 'response'] = result.get('response', '')
                    df.at[idx, 'model'] = model_config.id
            
            output_path = file_path.parent / f"{file_path.stem}_{model_config.id}_results.tsv"
            df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Results saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
