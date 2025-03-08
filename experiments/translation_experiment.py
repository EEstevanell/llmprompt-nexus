import asyncio
import yaml
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

from src.core.framework import LLMFramework
from templates.defaults import templates as default_templates
from src.models.registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TranslationExperiment:
    def __init__(self, experiment_path: Path, model_id: str, batch_size: int = 10):
        self.experiment_path = experiment_path
        self.model_id = model_id
        self.batch_size = batch_size
        
        # Get model configuration from registry
        self.model_config = registry.get_model(model_id)
        if not self.model_config:
            raise ValueError(f"Model {model_id} not found in registry")
            
        self.framework = None
        self.config = None
        self.data = None
        
    async def initialize(self, api_keys: Dict[str, str]) -> None:
        """Initialize the experiment with configuration and framework"""
        try:
            # Load experiment config
            with open(self.experiment_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            if not self.config.get('source_language') or not self.config.get('target_languages'):
                raise ValueError("Experiment config must specify source_language and target_languages")
                
            # Load translation data
            data_path = self.experiment_path.parent / 'data.csv'
            if not data_path.exists():
                raise FileNotFoundError(f"Translation data not found at {data_path}")
                
            self.data = pd.read_csv(data_path)
            if 'source_text' not in self.data.columns:
                raise ValueError("Data must contain 'source_text' column")
            
            # Initialize framework
            self.framework = LLMFramework(api_keys)
            logger.info(f"Initialized experiment with model {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize experiment: {str(e)}")
            raise
            
    async def process_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """Process a batch of texts for translation"""
        try:
            source_lang = self.config['source_language']
            
            # Create a temporary dataframe for batch processing
            batch_df = pd.DataFrame({'source_text': texts})
            
            # Add required fields for template formatting
            batch_df['source_lang'] = source_lang
            batch_df['target_lang'] = target_lang
            batch_df['intention'] = 'translate'  # Use the translate template
            
            # Create a temporary file for the batch
            batch_file = self.experiment_path.parent / f"temp_batch_{self.model_id}.tsv"
            batch_df.to_csv(batch_file, sep='\t', index=False)
            
            # Use the framework to process the file
            # The framework will handle batch processing internally
            await self.framework.process_file(
                file_path=batch_file,
                model_config=self.model_config,
                templates=default_templates,
                batch_mode=True,
                batch_size=self.batch_size,
                max_concurrent=5
            )
            
            # Read results back
            result_file = batch_file.parent / f"{batch_file.stem}_{self.model_id}_results.tsv"
            if result_file.exists():
                results_df = pd.read_csv(result_file, sep='\t')
                translations = results_df['response'].tolist()
                
                # Clean up temporary files
                try:
                    batch_file.unlink()
                    result_file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete temporary files: {str(e)}")
                    
                return translations
            else:
                raise FileNotFoundError(f"Results file not found at {result_file}")
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
            
    async def run(self) -> None:
        """Run the translation experiment"""
        try:
            source_lang = self.config['source_language']
            target_languages = self.config['target_languages']
            results = []
            
            # Process each target language
            for target_lang in target_languages:
                logger.info(f"Processing translations from {source_lang} to {target_lang}")
                
                # Process in batches
                for i in range(0, len(self.data), self.batch_size):
                    batch = self.data.iloc[i:i+self.batch_size]
                    source_texts = batch['source_text'].tolist()
                    
                    logger.info(f"Processing batch {i//self.batch_size + 1} with {len(source_texts)} texts")
                    translations = await self.process_batch(source_texts, target_lang)
                    
                    # Store results
                    for source, translation in zip(source_texts, translations):
                        results.append({
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'source_text': source,
                            'translation': translation,
                            'model': self.model_id
                        })
                        
            # Save results
            results_df = pd.DataFrame(results)
            output_path = self.experiment_path.parent / f'results_{self.model_id}.csv'
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}")
            raise