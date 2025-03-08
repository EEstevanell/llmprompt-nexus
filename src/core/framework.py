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
from src.core.client_factory import create_client
from src.processors.base import BaseProcessor, BatchVariableProvider, SimpleBatchVariableProvider
from src.processors.factory import create_processor
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
        self.api_keys = api_keys
        self.template_manager = get_default_template_manager()
    
    def get_client(self, api_name: str):
        """Get or create an API client instance."""
        if api_name not in self.api_keys:
            raise ValueError(f"No API key provided for {api_name}")
        return create_client(api_name, self.api_keys[api_name])
    
    async def generate(
        self,
        prompt: str,
        model_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using a specific model."""
        model_config = registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        processor = create_processor(client, model_config)
        
        return await processor.process_item({
            'prompt': prompt,
            'model': model_config.name,
            **kwargs
        })
    
    async def generate_batch(
        self,
        prompts: List[str],
        model_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts in parallel."""
        model_config = registry.get_model(model_id)
        client = self.get_client(model_config.provider)
        processor = create_processor(client, model_config)
        
        # Create batch items
        items = [
            {'prompt': prompt, 'model': model_config.name, **kwargs}
            for prompt in prompts
        ]
        
        # Process batch using client's parallel processing
        return await processor.process_batch(items, model_config.name)
    
    async def process_with_template(
        self,
        input_data: Dict[str, Any],
        model_id: str,
        template: Template,
        global_vars: Optional[Dict[str, Any]] = None,
        transform_input: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process a single input using a template."""
        # Transform input if needed
        if transform_input:
            variables = transform_input(input_data)
        else:
            variables = input_data
            
        # Add global variables
        if global_vars:
            variables.update(global_vars)
            
        # Render template
        prompt = template.render(variables)
        
        # Generate response
        return await self.generate(prompt, model_id)
    
    async def process_batch_with_template(
        self,
        inputs: List[Dict[str, Any]],
        model_id: str,
        template: Template,
        global_vars: Optional[Dict[str, Any]] = None,
        transform_input: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple inputs using a template in parallel."""
        # Transform inputs and prepare prompts
        prompts = []
        for input_data in inputs:
            # Transform input if needed
            if transform_input:
                variables = transform_input(input_data)
            else:
                variables = input_data.copy()
                
            # Add global variables
            if global_vars:
                variables.update(global_vars)
                
            # Render template
            prompt = template.render(variables)
            prompts.append(prompt)
        
        # Generate responses in parallel
        return await self.generate_batch(prompts, model_id)
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_id: str,
        **kwargs
    ) -> str:
        """Translate text using a language model."""
        template = self.template_manager.get_template("translate")
        
        result = await self.process_with_template(
            input_data={'text': text},
            model_id=model_id,
            template=template,
            global_vars={
                'source_language': source_lang,
                'target_language': target_lang
            },
            **kwargs
        )
        
        return result.get('response', '')
    
    async def translate_batch(
        self,
        texts: List[Dict[str, str]],
        model_id: str,
        batch_size: int = 10
    ) -> List[Dict[str, str]]:
        """
        Translate a batch of texts using the specified model with default translation template.
        For custom translation templates, use run_with_model instead.
        
        Args:
            texts (List[Dict]): List of dictionaries with 'text', 'source_lang', and 'target_lang' keys
            model_id (str): The model ID to use for translation
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 10.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with translation results
        """
        # Always use default translation template
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
            client = self.get_client(model_config.api)
            processor = create_processor(
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
    
    async def process_batch_with_template(
        self,
        inputs: List[Dict[str, Any]],
        model_id: str,
        template: Template,
        global_vars: Optional[Dict[str, Any]] = None,
        transform_input: Optional[callable] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Simplified interface for batch processing with templates.
        
        Args:
            inputs: List of input dictionaries to process
            model_id: The model ID to use
            template: Template to use for formatting
            global_vars: Variables that apply to all examples in the batch
            transform_input: Optional function to transform each input before template rendering.
                           Function signature: (dict) -> dict
            batch_size: Size of batches for processing
            
        Returns:
            List of results from the model
        """
        if transform_input is None:
            # Default transform just returns the input as is
            transform_input = lambda x: x
            
        # Transform each input and add global variables
        processed_inputs = []
        for input_data in inputs:
            # Transform the input first
            transformed = transform_input(input_data)
            
            # Add global variables if any
            if global_vars:
                transformed = {**global_vars, **transformed}
                
            processed_inputs.append(transformed)
            
        return await self.run_with_model(
            input_data=processed_inputs,
            model_id=model_id,
            template=template,
            batch_mode=True
        )

    async def run_with_model(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        model_id: str,
        template: Optional[Template] = None,
        batch_mode: bool = False,
        variable_mapping: Optional[Dict[str, str]] = None,
        batch_variable_provider: Optional[BatchVariableProvider] = None,
        global_variables: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process input data with a specific model using a template.
        
        This is the main generic interface for processing any kind of input
        with any supported model. For simpler batch processing with templates,
        consider using process_batch_with_template instead.
        
        Args:
            input_data: Single dictionary or list of dictionaries containing input data
            model_id: The model ID to use for processing
            template: Template instance to use for formatting the prompt
            batch_mode: Whether to process as a batch (if input_data is a list)
            variable_mapping: Optional mapping from source variable names to template variable names
            batch_variable_provider: Optional provider for handling global and per-example variables
            global_variables: Optional dictionary of variables that apply to all examples
            
        Returns:
            Processed results from the model
        """
        # Get model configuration
        model_config = registry.get_model(model_id)
        
        # If no batch variable provider but we have global vars or mapping, create a simple provider
        if batch_variable_provider is None and (global_variables or variable_mapping):
            batch_variable_provider = SimpleBatchVariableProvider(
                global_vars=global_variables or {},
                variable_mapping=variable_mapping
            )
            
        # Handle variable mapping through the provider if we have one
        if batch_variable_provider is not None:
            if batch_mode and isinstance(input_data, list):
                # Get global variables once
                global_vars = batch_variable_provider.get_global_variables()
                
                # Map each example with both global and example-specific variables
                input_data = [
                    {
                        **global_vars,
                        **batch_variable_provider.get_example_variables(item)
                    }
                    for item in input_data
                ]
            elif not batch_mode and isinstance(input_data, dict):
                # For single items, merge global and example variables
                input_data = {
                    **batch_variable_provider.get_global_variables(),
                    **batch_variable_provider.get_example_variables(input_data)
                }
                
        # For backward compatibility, handle old-style variable mapping
        elif variable_mapping is not None:
            if batch_mode and isinstance(input_data, list):
                input_data = [
                    {variable_mapping.get(k, k): v for k, v in item.items()}
                    for item in input_data
                ]
            elif not batch_mode and isinstance(input_data, dict):
                input_data = {
                    variable_mapping.get(k, k): v 
                    for k, v in input_data.items()
                }
        
        # Get client and processor
        client = self.get_client(model_config.api)
        processor = create_processor(
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
