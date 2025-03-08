#!/usr/bin/env python
# main.py - Example usage of UnifiedLLM framework for translation

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.core.framework import UnifiedLLM
from src.templates.base import Template
from src.templates.defaults import templates as default_templates
from src.utils.logger import get_logger, VerboseLevel
from src.templates.defaults import get_template_manager, render_template

logger = get_logger(__name__)

async def process_parallel_batch(framework: UnifiedLLM, batch_texts: List[Dict], model_id: str, 
                               template: Template, global_vars: Dict[str, Any], 
                               transform_input: callable, max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Process batch items in parallel with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(batch_texts)
    
    async def process_item(index: int, item: Dict):
        async with semaphore:
            try:
                result = await framework.process_batch_with_template(
                    inputs=[item],
                    model_id=model_id,
                    template=template,
                    global_vars=global_vars,
                    transform_input=transform_input
                )
                results[index] = result[0]
            except Exception as e:
                logger.error(f"Error processing batch item {index}: {str(e)}")
                results[index] = {"error": str(e)}
    
    tasks = [process_item(i, item) for i, item in enumerate(batch_texts)]
    await asyncio.gather(*tasks)
    return results

async def main():
    try:
        # Load API keys from environment variables
        api_keys = {
            "perplexity": os.getenv("PERPLEXITY_API_KEY") or "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",
            "openai": os.getenv("OPENAI_API_KEY")
        }
        
        # Example texts for translation
        texts = [
            "The artificial intelligence revolution has transformed how we interact with technology.",
            "Machine learning models can now understand and generate human language with impressive accuracy."
        ]
        
        # Models to use
        models_to_run = ["sonar", "sonar-pro"]
        
        logger.info("Translation example with UnifiedLLM framework")
        
        # Initialize framework and template manager
        framework = UnifiedLLM(api_keys)
        template_manager = get_template_manager()
        
        # 1. Simple translation using built-in templates
        logger.info("\n--- Example 1: Simple Translation with Built-in Templates ---")
        tasks = []
        for text in texts:
            logger.info(f"Original text (English): {text}")
            
            for model_id in models_to_run:
                async def translate_text():
                    try:
                        result = await framework.translate(
                            text=text,
                            source_lang="English",
                            target_lang="Spanish",
                            model_id=model_id
                        )
                        logger.info(f"[{model_id}] Translation to Spanish: {result}")
                    except Exception as e:
                        logger.error(f"Error translating with {model_id}: {str(e)}")
                
                tasks.append(translate_text())
        
        # Run translations in parallel
        await asyncio.gather(*tasks)

        # 2. Translation with custom template using parallel batch processing
        logger.info("\n--- Example 2: Translation with Custom Template (Parallel Batch) ---")
        
        # Register a custom template with different variable names
        custom_template = """
        Please act as an expert translator and translate this content:
        
        Input Language: {input_language}
        Output Language: {output_language}
        Preserve style: {preserve_style}
        
        Content to translate:
        {input_text}
        
        Translation guidelines:
        - Maintain the original tone and meaning
        - Preserve any technical terminology
        - Ensure natural flow in the target language
        """
        
        template_manager.register_template(Template(
            template_text=custom_template,
            name="custom_translate",
            description="Custom translation template with different variable names"
        ))

        # Create batch of texts
        batch_texts = [{'text': text} for text in texts * 2]  # Multiply texts for demonstration
        
        # Define a simple transform function
        def transform_input(input_dict: Dict) -> Dict:
            """Transform input dictionary to match template variables."""
            return {
                'input_text': input_dict['text']
            }
        
        # Process batch in parallel with rate limiting
        batch_results = await process_parallel_batch(
            framework=framework,
            batch_texts=batch_texts,
            model_id="sonar-pro",
            template=template_manager.get_template("custom_translate"),
            global_vars={
                'input_language': 'English',
                'output_language': 'Spanish',
                'preserve_style': 'yes'
            },
            transform_input=transform_input,
            max_concurrent=5  # Adjust based on rate limits
        )
        
        for i, result in enumerate(batch_results):
            logger.info(f"\nBatch item {i+1}:")
            logger.info(f"Original: {batch_texts[i]['text']}")
            logger.info(f"Translation: {result.get('response', 'Error')}")

        # 3. Translation with advanced formatting options
        logger.info("\n--- Example 3: Translation with Advanced Formatting ---")
        
        format_tasks = []
        for fmt in ['normal', 'uppercase', 'friendly']:
            async def process_format(format_type):
                logger.info(f"\nProcessing with {format_type} formatting...")
                
                def make_transform(f_type):
                    def transform(input_dict):
                        text = input_dict['text']
                        if f_type == 'uppercase':
                            text = text.upper()
                        elif f_type == 'friendly':
                            text = f"Hey! Here's what I'd like to translate: {text}"
                        return {'input_text': text}
                    return transform
                
                results = await process_parallel_batch(
                    framework=framework,
                    batch_texts=batch_texts[:1],
                    model_id="sonar-pro",
                    template=template_manager.get_template("custom_translate"),
                    global_vars={
                        'input_language': 'English',
                        'output_language': 'Spanish',
                        'preserve_style': 'yes'
                    },
                    transform_input=make_transform(format_type),
                    max_concurrent=3
                )
                
                for i, result in enumerate(results):
                    logger.info(f"Original ({format_type}): {batch_texts[i]['text']}")
                    logger.info(f"Translation: {result.get('response', 'Error')}")
            
            format_tasks.append(process_format(fmt))
        
        # Run all format variations in parallel
        await asyncio.gather(*format_tasks)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
