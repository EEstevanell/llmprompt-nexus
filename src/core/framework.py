"""
UnifiedLLM Framework - A unified interface for interacting with various LLM providers.

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
from src.core.client_factory import ClientFactory
from src.processors.base import BaseProcessor
from src.processors.factory import ProcessorFactory
from src.models.model_config import ModelConfig
from src.models.registry import registry
from src.utils.logger import get_logger, VerboseLevel
from src.templates.base import Template
from src.templates.defaults import get_template_manager as get_default_template_manager

logger = get_logger(__name__)

class UnifiedLLM:
    """
    A unified framework for interacting with various LLM providers through a consistent interface.
    
    This framework provides:
    - Multi-provider support through a unified interface
    - Template-based interactions for different use cases
    - Built-in rate limiting and API key management
    - Batch processing capabilities are supported
    - Asynchronous processing
    
    Example:
    ```
        api_keys = {
            "openai": "sk-...",
            "perplexity": "pplx-..."
        }
        
        framework = UnifiedLLM(api_keys)
        result = await framework.run_with_model(
            input_data={"text": "Hello", "intention": "sentiment"},
            model_id="gpt-4"
        )
    ```
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the framework with API keys for different providers.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys
        """
        self._validate_api_keys(api_keys)
        self.api_keys = api_keys
        self.client_factory = ClientFactory(api_keys)
        self.processor_factory = ProcessorFactory()
        logger.info("UnifiedLLM framework initialized")
    
    def _validate_api_keys(self, api_keys: Dict[str, Optional[str]]) -> None:
        """Validate that required API keys are present based on available providers."""
        if not api_keys:
            raise ValueError("No API keys provided")
        
        # Get list of providers from registry
        providers = registry.list_providers()
        missing_keys = [provider for provider in providers if provider in api_keys and not api_keys[provider]]
        
        if missing_keys:
            logger.warning(f"Missing API keys for providers: {', '.join(missing_keys)}")
    
    def get_model_configs(self, models_to_run: List[str]) -> List[ModelConfig]:
        """
        Validate models and return their configurations.
        
        Args:
            models_to_run: List of model IDs to validate
            
        Returns:
            List of validated ModelConfig instances
            
        Raises:
            ValueError: If a model ID is invalid or its provider's API key is missing
        """
        model_configs = []
        for model_id in models_to_run:
            try:
                # The registry.get_model already validates if the model exists
                model_config = registry.get_model(model_id)
                
                # Make sure we have the API key for this model's provider
                if model_config.provider not in self.api_keys or not self.api_keys[model_config.provider]:
                    raise ValueError(f"Missing API key for provider '{model_config.provider}' required by model '{model_id}'")
                
                model_configs.append(model_config)
            except Exception as e:
                logger.error(f"Error validating model {model_id}: {str(e)}")
                raise
        return model_configs
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_id: str,
        template: Optional[Template] = None
    ) -> str:
        """
        High-level method to translate text using the specified model.
        
        Args:
            text (str): The text to translate
            source_lang (str): Source language code (e.g., 'en')
            target_lang (str): Target language code (e.g., 'es')
            model_id (str): The model ID to use for translation
            template (Template, optional): Custom template to use
            
        Returns:
            str: The translated text
        """
        # Use default translation template if none provided
        if template is None:
            template_manager = get_default_template_manager()
            template = template_manager.get_template('translate')
        
        # Create input data
        input_data = {
            'text': text,
            'source_lang': source_lang,
            'target_lang': target_lang
        }
        
        result = await self.run_with_model(
            input_data=input_data,
            model_id=model_id,
            template=template
        )
        
        return result.get('response', '')
    
    async def translate_batch(
        self,
        texts: List[Dict[str, str]],
        model_id: str,
        template: Optional[Template] = None,
        batch_size: int = 10
    ) -> List[Dict[str, str]]:
        """
        Translate a batch of texts using the specified model.
        
        Args:
            texts (List[Dict]): List of dictionaries with 'text', 'source_lang', and 'target_lang' keys
            model_id (str): The model ID to use for translation
            template (Template, optional): Custom template to use
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 10.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with translation results
        """
        # Use default translation template if none provided
        if template is None:
            template_manager = get_default_template_manager()
            template = template_manager.get_template('translate')
        
        return await self.run_with_model(
            input_data=texts,
            model_id=model_id,
            template=template,
            batch_mode=True
        )
    
    async def process_file(
        self,
        file_path: Path,
        model_config: ModelConfig,
        template: Optional[Template] = None,
        batch_mode: bool = False,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> Path:
        """
        Process a file with the specified model configuration.
        
        Args:
            file_path: Path to the input file (TSV format)
            model_config: Configuration for the model to use
            template: Optional Template instance to use for formatting
            batch_mode: Whether to use batch processing
            batch_size: Size of batches when batch_mode is True
            max_concurrent: Maximum number of concurrent tasks when not in batch mode
            
        Returns:
            Path to the output file with results
            
        Raises:
            ValueError: If the file cannot be read or processed
            Exception: If there are errors during processing
        """
        logger.info(f"Processing file {file_path} with model {model_config.id}")
        logger.debug(f"Batch mode: {batch_mode}, batch size: {batch_size}, max concurrent: {max_concurrent}")
        
        try:
            # Read the file
            df = pd.read_csv(file_path, sep='\t')
            if df.empty:
                raise ValueError(f"Input file {file_path} is empty")
            
            logger.info(f"Read {len(df)} rows from {file_path}")

            # Get appropriate client and processor
            client = self.client_factory.get_client(model_config.api)
            processor = self.processor_factory.get_processor(
                model_config.api, 
                client, 
                model_config,
                template
            )
            
            # Process the data
            if batch_mode:
                await self._process_batch(processor, df, model_config, batch_size)
            else:
                await self._process_sequential(processor, df, model_config, max_concurrent)
                
            # Save results
            output_path = file_path.parent / f"{file_path.stem}_{model_config.id}_results.tsv"
            df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Results saved to {output_path}")
            
            return output_path
            
        except pd.errors.EmptyDataError:
            logger.error(f"Input file {file_path} is empty")
            raise ValueError(f"Input file {file_path} is empty")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise ValueError(f"Error parsing file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def _process_batch(self, processor: BaseProcessor, df: pd.DataFrame, model_config: ModelConfig, batch_size: int):
        """Process data in batch mode."""
        rows = df.to_dict('records')
        total_batches = (len(rows) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(rows)} rows in {total_batches} batches")
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                results = await processor.process_batch(batch)
                
                # Update DataFrame with results
                for j, result in enumerate(results):
                    idx = i + j
                    if idx < len(df):
                        df.at[idx, 'response'] = result.get('response', '')
                        df.at[idx, 'model'] = model_config.id
                        
                logger.debug(f"Completed batch {batch_num}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                raise

    async def _process_sequential(self, processor: BaseProcessor, df: pd.DataFrame, model_config: ModelConfig, max_concurrent: int):
        """Process data sequentially or with limited concurrency."""
        rows = df.to_dict('records')
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Processing {len(rows)} rows sequentially with max {max_concurrent} concurrent tasks")
        
        async def process_row(row, index):
            async with semaphore:
                try:
                    result = await processor.process_item(row)
                    df.at[index, 'response'] = result.get('response', '')
                    df.at[index, 'model'] = model_config.id
                    logger.debug(f"Processed row {index + 1}/{len(rows)}")
                except Exception as e:
                    logger.error(f"Error processing row {index}: {str(e)}")
                    raise
                
        tasks = [process_row(row, i) for i, row in enumerate(rows)]
        await asyncio.gather(*tasks)
    
    async def run_with_model(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        model_id: str,
        template: Optional[Template] = None,
        batch_mode: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process input data with a specific model using a template.
        
        This is the main generic interface for processing any kind of input
        with any supported model. The template defines how the input is formatted
        for the model.
        
        Args:
            input_data: Single dictionary or list of dictionaries containing input data
            model_id: The model ID to use for processing
            template: Template instance to use for formatting the prompt
            batch_mode: Whether to process as a batch (if input_data is a list)
            
        Returns:
            Processed results from the model
        """
        # Get model configuration
        model_config = registry.get_model(model_id)
        
        # Get client and processor
        client = self.client_factory.get_client(model_config.api)
        processor = self.processor_factory.get_processor(
            model_config.api,
            client,
            model_config,
            template
        )
        
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
            