# main.py

import asyncio
import os
from pathlib import Path

from src.core.framework import LLMFramework
from src.models.registry import registry
from src.templates.intention import templates

async def main():
    # Load API keys from environment variables
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }
    
    # Create framework instance
    framework = LLMFramework(api_keys)
    input_dir = Path("inputs")
    
    # Select which models to run
    models_to_run = ["gpt-4o-mini", "sonar-small", "sonar-huge"]
    
    for model_id in models_to_run:
        model_config = registry.get_model(model_id)
        
        # Determine if model supports batch processing
        supports_batch = model_config.api == "openai"
        
        for file_path in input_dir.glob("*.tsv"):
            print(f"Processing {file_path} with model {model_id}")
            
            await framework.process_file(
                file_path=file_path,
                model_config=model_config,
                templates=templates,
                batch_mode=supports_batch,  # Use batch mode for OpenAI
                batch_size=10,
                max_concurrent=5
            )

if __name__ == "__main__":
    asyncio.run(main())
