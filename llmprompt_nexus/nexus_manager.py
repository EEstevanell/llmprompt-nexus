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
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any, Union

from llmprompt_nexus.clients.client_factory import create_client
from llmprompt_nexus.processors.base import BaseProcessor
from llmprompt_nexus.processors.factory import create_processor
from llmprompt_nexus.models.model_config import ModelConfig
from llmprompt_nexus.models.registry import registry as model_registry
from llmprompt_nexus.templates import load_template, get_template_manager
from llmprompt_nexus.utils.logger import get_logger, configure_logger
from llmprompt_nexus.utils.progress import BatchProgressTracker

class NexusManager:
    """
    A unified framework for interacting with various LLM providers through a consistent interface.
    
    This framework provides:
    - Multi-provider support through a unified interface
    - Template-based interactions for different use cases
    - Built-in rate limiting and API key management
    - Batch processing capabilities with progress tracking
    - Asynchronous processing
    """
    
    def __init__(self, api_keys: Dict[str, str], log_level: str = "INFO"):
        """
        Initialize the framework with API keys for different providers.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys
            log_level: Logging level for the framework (DEBUG, INFO, WARNING, ERROR)
        """
        self.api_keys = api_keys
        
        # Configure logging for the entire framework
        configure_logger(log_level)
        self.logger = get_logger(__name__)
        
        # Load model registry only once during initialization
        self._model_registry = model_registry
        self._model_registry.load_models()
        self.logger.debug("Model registry loaded")
    
    def get_client(self, api_name: str):
        """Get or create an API client instance."""
        if api_name not in self.api_keys:
            raise ValueError(f"No API key provided for {api_name}")
        return create_client(api_name, self.api_keys[api_name])

    def _prepare_input_data(self, 
                           input_data: Union[str, Dict[str, Any]],
                           **kwargs) -> Dict[str, Any]:
        """
        Prepare input data from various formats.
        
        If input_data is a string, it's treated as a simple prompt.
        If it's a dictionary, it's used as-is with any additional kwargs.
        """
        if isinstance(input_data, str):
            # Handle simple string prompt
            prepared_data = {'prompt': input_data}
        elif isinstance(input_data, dict):
            # Use dictionary as-is
            prepared_data = input_data.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Add any additional kwargs
        prepared_data.update(**kwargs)
        return prepared_data
    
    def _get_template(self, 
                     template_name: Optional[str] = None,
                     template_config: Optional[Dict[str, Any]] = None):
        """
        Get a template based on name or config, defaulting to the basic template.
        """
        if template_config:
            # Custom template configuration provided
            return load_template("custom_template", template_config)
        elif template_name:
            # Named template
            try:
                tm = get_template_manager(template_name)
                return tm.get_template(template_name)
            except Exception as e:
                self.logger.warning(f"Failed to load named template '{template_name}': {str(e)}")
                self.logger.warning("Falling back to default template")
                
        # Default template for handling simple prompts
        try:
            tm = get_template_manager('default')
            return tm.get_template('default')
        except Exception as e:
            self.logger.warning(f"Failed to load default template: {str(e)}")
            # Create an inline minimal template as last resort
            return load_template("minimal", {
                "template": "{prompt}",
                "description": "Minimal inline template",
                "system_message": "You are a helpful AI assistant."
            })
    
    async def generate(self, 
                     input_data: Union[str, Dict[str, Any]],
                     model_id: str,
                     template_name: Optional[str] = None,
                     template_config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Process a single input with a model.
        
        Args:
            input_data: Either a prompt string or a dictionary of input data
            model_id: The model ID to use for processing
            template_name: Optional name of template to load
            template_config: Optional template configuration dictionary
            **kwargs: Additional parameters to pass to the processor
            
        Returns:
            Dictionary containing the model's response and metadata
        """
        # Get model configuration and client
        model_config = self._model_registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        
        # Get template (always using a template, with default as fallback)
        template = self._get_template(template_name, template_config)
        
        # Prepare input data
        prepared_data = self._prepare_input_data(input_data, model=model_config.name, **kwargs)
        
        # Create processor with the template
        processor = create_processor(client, model_config, template)
        
        try:
            # Process the item
            self.logger.debug(f"Processing single input with model {model_id}")
            return await processor.process_item(prepared_data)
        except Exception as e:
            self.logger.error(f"Error processing with model {model_id}: {str(e)}")
            raise
    
    async def generate_batch(self,
                           inputs: Union[List[str], List[Dict[str, Any]]],
                           model_id: str,
                           template_name: Optional[str] = None,
                           template_config: Optional[Dict[str, Any]] = None,
                           global_vars: Optional[Dict[str, Any]] = None,
                           silent: bool = False,
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple inputs with a model.
        
        The batch will be automatically managed based on model rate limits.
        
        Args:
            inputs: List of prompt strings or dictionaries
            model_id: The model ID to use for processing
            template_name: Optional name of template to load
            template_config: Optional template configuration dictionary
            global_vars: Optional dictionary of variables that apply to all items in the batch
            silent: If True, suppress progress bar (defaults to False)
            **kwargs: Additional parameters to pass to the processor
            
        Returns:
            List of dictionaries containing the model's responses and metadata
        """
        if not inputs:
            return []
            
        # Get model configuration and client
        model_config = self._model_registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        
        # Get template (always using a template, with default as fallback)
        template = self._get_template(template_name, template_config)
        
        # Prepare batch input data
        batch_data = [self._prepare_input_data(item, model=model_config.name, **kwargs) 
                     for item in inputs]
        
        # Create processor with the template
        processor = create_processor(client, model_config, template)
        
        try:
            # Set up progress tracking
            desc = f"Processing with {model_id}"
            with BatchProgressTracker(len(batch_data), desc=desc, silent=silent) as tracker:
                self.logger.debug(f"Processing batch of {len(inputs)} inputs with model {model_id}")
                # Process the batch with automatic queue management
                return await processor.process_batch(
                    batch_data, 
                    global_vars=global_vars,
                    progress_callback=tracker.update
                )
        except Exception as e:
            self.logger.error(f"Error processing batch with model {model_id}: {str(e)}")
            raise
    
    async def process_file(self,
                          file_path: Path,
                          model_id: str,
                          template_name: Optional[str] = None,
                          template_config: Optional[Dict[str, Any]] = None,
                          silent: bool = False,
                          **kwargs) -> Path:
        """
        Process a TSV file with a model.
        
        Args:
            file_path: Path to the input TSV file
            model_id: The model ID to use for processing
            template_name: Optional name of template to load
            template_config: Optional template configuration dictionary
            silent: If True, suppress progress bar (defaults to False)
            **kwargs: Additional parameters to pass to the processor
        
        Returns:
            Path to the output TSV file with results
        """
        # Get model configuration
        model_config = self._model_registry.get_model(model_id)
        
        self.logger.info(f"Processing file {file_path} with model {model_id}")
        
        # Get template (always using a template, with default as fallback)
        template = self._get_template(template_name, template_config)

        try:
            # Read the TSV file
            df = pd.read_csv(file_path, sep='\t')
            if df.empty:
                raise ValueError(f"Input file {file_path} is empty")
            
            self.logger.debug(f"Read {len(df)} rows from {file_path}")
            
            # Get client and create processor
            client = self.get_client(model_config.provider)
            processor = create_processor(client, model_config, template)
            
            # Always use batch processing for files - convert DataFrame to list of dictionaries
            items = df.to_dict('records')
            
            # Add any additional kwargs to each item
            for item in items:
                item.update(**kwargs)
            
            # Set up progress tracking
            desc = f"Processing {file_path.name} with {model_id}"
            with BatchProgressTracker(len(items), desc=desc, silent=silent) as tracker:
                # Use the smart batch processing system
                results = await processor.process_batch(
                    items,
                    progress_callback=tracker.update
                )
            
            # Update the DataFrame with results
            for idx, result in enumerate(results):
                df.at[idx, 'response'] = result.get('response', '')
                df.at[idx, 'model'] = model_id
            
            # Save results to a new TSV file
            output_path = file_path.parent / f"{file_path.stem}_{model_id}_results.tsv"
            df.to_csv(output_path, sep='\t', index=False)
            self.logger.info(f"Results saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path} with model {model_id}: {str(e)}")
            raise
