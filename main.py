# main.py
import asyncio
import os
from pathlib import Path
from utils.async_utils import ProcessingManager
from models.registry import registry
from templates.intention import templates

async def main():
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY") or "Perplexity API Key",
        "openai": os.getenv("OPENAI_API_KEY") or "OPENAI API Key"
    }
    
    manager = ProcessingManager(api_keys)
    input_dir = Path("inputs")
    
    # Select which models to run
    models_to_run = ["gpt-4o-mini", "sonar-small", "sonar-huge"]
    
    for model_id in models_to_run:
        model_config = registry.get_model(model_id)
        for file_path in input_dir.glob("*.tsv"):
            await manager.process_file(file_path, templates, model_config)

if __name__ == "__main__":
    asyncio.run(main())
