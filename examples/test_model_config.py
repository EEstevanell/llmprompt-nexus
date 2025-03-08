#!/usr/bin/env python
import asyncio
import os
from typing import Dict

from src.core.framework import UnifiedLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Example templates for different tasks
templates = {
    "summarize": {
        "template": "Please provide a concise summary of the following text:\n{text}"
    },
    "sentiment": {
        "template": "What is the sentiment of this text? Answer with POSITIVE, NEGATIVE, or NEUTRAL:\n{text}"
    },
    "classify": {
        "template": "Classify this text into one of these categories: TECH, BUSINESS, SCIENCE, OTHER:\n{text}"
    },
    "extract": {
        "template": "Extract key entities (people, organizations, locations) from this text:\n{text}"
    }
}

async def demonstrate_capabilities():
    # Load API keys from environment
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

    # Initialize framework
    framework = UnifiedLLM(api_keys)

    # Example texts
    texts = [
        "Apple announced its new Vision Pro headset, which will retail for $3499.",
        "Climate change poses significant risks to coastal communities worldwide.",
        "The company's stock dropped 5% after disappointing quarterly results.",
        "The AI conference brought together researchers from MIT, Google, and Stanford."
    ]

    # Test different tasks with different models
    models = ["gpt-4", "sonar"]
    tasks = ["summarize", "sentiment", "classify", "extract"]

    for model_id in models:
        logger.info(f"\nTesting model: {model_id}")
        
        # Single item processing
        for task, text in zip(tasks, texts):
            try:
                result = await framework.run_with_model(
                    input_data={"text": text},
                    model_id=model_id,
                    templates={task: templates[task]}
                )
                logger.info(f"\n{task.upper()} TASK:")
                logger.info(f"Input: {text}")
                logger.info(f"Output: {result.get('response', '')}")
            except Exception as e:
                logger.error(f"Error with {task} task: {str(e)}")

        # Batch processing example
        try:
            batch_items = [
                {"text": text, "task": task}
                for text, task in zip(texts, tasks)
            ]
            
            results = await framework.run_with_model(
                input_data=batch_items,
                model_id=model_id,
                templates=templates,
                batch_mode=True
            )
            
            logger.info("\nBATCH PROCESSING RESULTS:")
            for item, result in zip(batch_items, results):
                logger.info(f"\nTask: {item['task']}")
                logger.info(f"Input: {item['text']}")
                logger.info(f"Output: {result.get('response', '')}")
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demonstrate_capabilities())