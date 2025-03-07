import asyncio
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

from src.core.client_factory import ClientFactory
from src.processors.base_processor import BaseProcessor
from src.processors.processor_factory import ProcessorFactory
from src.models.model_config import ModelConfig
from src.utils.logger import get_logger, VerboseLevel

logger = get_logger(__name__)

class LLMFramework:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.client_factory = ClientFactory(api_keys)
        self.processor_factory = ProcessorFactory()
        logger.info("LLMFramework initialized")

    async def process_file(
        self,
        file_path: Path,
        model_config: ModelConfig,
        templates: Dict,
        batch_mode: bool = False,
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """Process a file with the specified model configuration."""
        logger.info(f"Processing file {file_path} with model {model_config.id}")
        logger.debug(f"Batch mode: {batch_mode}, batch size: {batch_size}, max concurrent: {max_concurrent}")
        
        try:
            # Read the file
            df = pd.read_csv(file_path, sep='\t')
            logger.info(f"Read {len(df)} rows from {file_path}")

            # Get appropriate client
            client = self.client_factory.get_client(model_config.api)
            
            # Get appropriate processor
            processor = self.processor_factory.get_processor(
                model_config.api, 
                client, 
                model_config,
                templates
            )
            
            if batch_mode:
                await self._process_batch(processor, df, model_config, batch_size)
            else:
                await self._process_sequential(processor, df, model_config, max_concurrent)
                
            # Save results
            output_path = file_path.parent / f"{file_path.stem}_{model_config.id}_results.tsv"
            df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Results saved to {output_path}")
            
            return output_path
            
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
