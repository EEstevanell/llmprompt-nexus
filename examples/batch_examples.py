#!/usr/bin/env python
"""
Example script demonstrating batch processing capabilities of the UnifiedLLM framework.
Shows how to use both default and custom templates with batch processing.
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, List

from src.core.framework import NexusManager
from src.templates.defaults import get_template_manager
from src.templates.manager import TemplateManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Example custom template with system message
batch_translation_template = {
    "batch_translation": {
        "template": "Translate the following text from {source_language} to {target_language}:\n\n{text}",
        "description": "Optimized translation template for batch processing",
        "system_message": "You are an expert translator specializing in batch translation tasks. Focus on maintaining consistency across all translations while preserving meaning and context."
    }
}

async def demonstrate_default_template_batch():
    """Demonstrate batch processing using a default template."""
    logger.info("\n=== Batch Processing with Default Template ===")
    
    # Initialize framework with API keys
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }
    
    framework = NexusManager(api_keys)
    
    # Get default translation template
    tm = get_template_manager('translation')
    template = tm.get_template('translation')
    
    # Prepare batch of texts to translate
    texts = [
        "The morning sun cast long shadows across the field.",
        "Innovation drives progress in technology.",
        "Climate change requires global cooperation.",
        "Art reflects the cultural values of society.",
        "Education empowers future generations."
    ]
    
    # Create batch input data
    batch_inputs = [
        {
            "text": text,
            "source_language": "English",
            "target_language": "Spanish"
        }
        for text in texts
    ]
    
    # Process batch with progress tracking
    def progress_callback(status: str, progress: float, completed: int, failed: int, in_progress: int):
        logger.info(f"Batch Progress: {progress:.1f}% - Completed: {completed}, Failed: {failed}, In Progress: {in_progress}")
    
    results = await framework.run_batch_with_model(
        input_data=batch_inputs,
        model_id="sonar-pro",
        template=template,
        progress_callback=progress_callback
    )
    
    # Display results
    logger.info("\nBatch Translation Results:")
    for i, (text, result) in enumerate(zip(texts, results)):
        logger.info(f"\nOriginal ({i+1}): {text}")
        logger.info(f"Translation: {result.get('response', 'Error')}")

async def demonstrate_custom_template_batch():
    """Demonstrate batch processing using a custom template."""
    logger.info("\n=== Batch Processing with Custom Template ===")
    
    # Initialize framework
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }
    
    framework = NexusManager(api_keys)
    
    # Create template manager and load custom template
    tm = TemplateManager()
    tm.load_from_dict(batch_translation_template)
    template = tm.get_template('batch_translation')
    
    # Prepare technical texts for batch translation
    technical_texts = [
        "The API implements robust rate limiting mechanisms.",
        "Data structures are optimized for concurrent access.",
        "The framework supports asynchronous batch processing.",
        "Unit tests ensure code reliability.",
        "Configuration is managed through YAML files."
    ]
    
    # Create batch input data
    batch_inputs = [
        {
            "text": text,
            "source_language": "English",
            "target_language": "French"
        }
        for text in technical_texts
    ]
    
    # Process batch with custom metadata
    metadata = {
        "domain": "technical",
        "batch_type": "custom_template",
        "content_type": "documentation"
    }
    
    results = await framework.run_batch_with_model(
        input_data=batch_inputs,
        model_id="sonar-pro",
        template=template,
        metadata=metadata
    )
    
    # Display results
    logger.info("\nTechnical Batch Translation Results:")
    for i, (text, result) in enumerate(zip(technical_texts, results)):
        logger.info(f"\nOriginal ({i+1}): {text}")
        logger.info(f"Translation: {result.get('response', 'Error')}")

async def main():
    """Run batch processing examples."""
    try:
        await demonstrate_default_template_batch()
        await demonstrate_custom_template_batch()
    except Exception as e:
        logger.error(f"Error running batch examples: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())