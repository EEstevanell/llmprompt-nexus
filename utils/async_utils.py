# src/utils/async_utils.py
import asyncio
import aiohttp
from pathlib import Path
import pandas as pd
from typing import List, Dict
from models.config import ModelConfig
from clients.base import BaseClient
from clients.perplexity import PerplexityClient
from clients.openai import OpenAIClient
from tqdm import tqdm

class ProcessingManager:
    def __init__(self, api_keys: Dict[str, str]):
        self.clients = {
            "perplexity": PerplexityClient(api_keys["perplexity"]),
            "openai": OpenAIClient(api_keys["openai"])
        }
    
    def get_client(self, api_name: str) -> BaseClient:
        return self.clients[api_name]
    
    async def process_file(
        self,
        input_file: Path,
        templates: List[Dict],
        model_config: ModelConfig,
        batch_size: int = 10
    ):
        # Read TSV with only 'Text' column
        df = pd.read_csv(input_file, sep='\t')
        client = self.get_client(model_config.api)
        
        async with aiohttp.ClientSession() as session:
            for template in templates:
                column_name = f"GlobalIntention_{template['name']}_{model_config.name}"
                df[column_name] = ""
                
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    try:
                        data = self._prepare_request_data(model_config, template, row)
                        result = await client.make_request(session, model_config.name, data)
                        df.at[idx, column_name] = self._extract_response(result)
                        
                        if idx % batch_size == 0:
                            self._save_progress(df, input_file, model_config, temporary=True)
                    
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                        df.at[idx, column_name] = "ERROR"
        
        self._save_progress(df, input_file, model_config)
    
    def _prepare_request_data(self, model_config: ModelConfig, template: Dict, row: pd.Series) -> Dict:
        if model_config.api == "perplexity":
            return {
                "model": model_config.name,
                "messages": [
                    {"role": "system", "content": "Eres un experto lingüista español."},
                    {"role": "user", "content": f"{template['template']}\n{row['text']}"}
                ],
                "return_images": False,
                "return_related_questions": False,
                "stream": False,
                "temperature": model_config.parameters.get("temperature", 0.2),
                "max_tokens": model_config.parameters.get("max_tokens", 150),
            }
        elif model_config.api == "openai":
            return {
                "model": model_config.name,
                "messages": [
                    {"role": "system", "content": "Eres un experto lingüista español."},
                    {"role": "user", "content": f"{template['template']}\n{row['Text']}"}
                ],
                "temperature": model_config.parameters.get("temperature", 0.7),
                "max_tokens": model_config.parameters.get("max_tokens", 150),
            }
        else:
            raise ValueError(f"Unsupported API type: {model_config.api}")
    
    def _extract_response(self, result: Dict) -> str:
        try:
            response_text = result['choices'][0]['message']['content']
            # Extract text between brackets
            return response_text
        except (KeyError, IndexError) as e:
            print(f"Error extracting response: {e}")
            return "ERROR"
    
    def _save_progress(self, df: pd.DataFrame, input_file: Path, model_config: ModelConfig, temporary: bool = False):
        suffix = "_temp" if temporary else ""
        output_file = input_file.parent / f"{input_file.stem}_{model_config.name}{suffix}.csv"
        
        try:
            df.to_csv(output_file, index=False)
            if not temporary:
                # Remove temporary file if it exists when saving final version
                temp_file = input_file.parent / f"{input_file.stem}_{model_config.name}_temp.csv"
                if temp_file.exists():
                    temp_file.unlink()
        except Exception as e:
            print(f"Error saving progress to {output_file}: {e}")
