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


async def process_with_llm(
    text: str, 
    intention: str,
    model_id: str,
    api_keys: Dict[str, str],
    templates: Optional[Dict[str, str]] = None,
    field_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Process text with a language model using specified intention/template.
    
    Args:
        text (str): The text to process
        intention (str): The processing intention (defines template to use)
        model_id (str): The model ID to use
        api_keys (Dict[str, str]): Dictionary of API keys
        templates (Dict[str, str]): Optional custom templates to use
        field_mapping (Dict[str, str]): Optional mapping from input fields to template variables
        
    Returns:
        str: The processed text response
    """
    # Initialize framework
    framework = UnifiedLLM(api_keys)
    
    # Prepare input data with optional field mapping
    input_data = {'text': text, 'intention': intention}
    
    # Process the text
    result = await framework.run_with_model(
        input_data=input_data,
        model_id=model_id,
        templates=templates
    )
    
    return result.get('response', '')


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_id: str,
    api_keys: Dict[str, str],
    custom_template: Optional[str] = None,
    field_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Translate text using the framework with optional custom template.
    
    Args:
        text (str): The text to translate
        source_lang (str): Source language code (e.g., 'en')
        target_lang (str): Target language code (e.g., 'es')
        model_id (str): The model ID to use
        api_keys (Dict[str, str]): Dictionary of API keys
        custom_template (str, optional): Custom translation template
        field_mapping (Dict[str, str], optional): Custom field mapping
        
    Returns:
        str: The translated text
    """
    # Initialize framework and template manager
    framework = UnifiedLLM(api_keys)
    template_manager = get_template_manager()
    
    # Set up templates - either use custom or get built-in from manager
    if custom_template:
        template_manager.register_template(Template(
            template_text=custom_template,
            name="custom_translate",
            description="Custom translation template"
        ))
        templates = {'translate': custom_template}
    else:
        # Use the built-in translation template
        built_in_template = template_manager.get_template('translate')
        templates = {'translate': built_in_template.template_text}
    
    # Prepare input data with field mapping
    input_data = {
        'text': text,
        'source_lang': source_lang,
        'target_lang': target_lang
    }
    
    if field_mapping:
        input_data = template_manager.apply_field_mapping(input_data, field_mapping)
    
    # Process the translation
    result = await framework.run_with_model(
        input_data=input_data,
        model_id=model_id,
        templates=templates
    )
    
    return result.get('response', '')


async def process_batch(
    items: List[Dict[str, str]], 
    model_id: str,
    api_keys: Dict[str, str],
    templates: Optional[Dict[str, str]] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    batch_size: int = 10
) -> List[Dict]:
    """
    Process a batch of items with a language model.
    
    Args:
        items: List of dictionaries with text and intention
        model_id: The model ID to use
        api_keys: Dictionary of API keys
        templates: Optional custom templates to use
        field_mapping: Optional mapping from input fields to template variables
        batch_size: Size of batches for processing
    """
    framework = UnifiedLLM(api_keys)
    
    # Apply field mapping if provided
    if field_mapping:
        mapped_items = []
        for item in items:
            mapped_item = {}
            for dest_key, src_key in field_mapping.items():
                if src_key in item:
                    mapped_item[dest_key] = item[src_key]
            mapped_items.append(mapped_item)
        items = mapped_items
    
    return await framework.run_with_model(
        input_data=items,
        model_id=model_id,
        templates=templates,
        batch_mode=True
    )


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
        for text in texts:
            logger.info(f"Original text (English): {text}")
            
            for model_id in models_to_run:
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
        
        # 2. Translation with custom template
        logger.info("\n--- Example 2: Translation with Custom Template ---")
        
        # Register a custom template with the manager
        custom_template = """
        Please translate the following text:
        
        Original ({input_language}): 
        {input_text}
        
        Translate to {output_language}. 
        Make sure to maintain the original tone and meaning.
        """
        
        template_manager.register_template(Template(
            template_text=custom_template,
            name="custom_translate",
            description="Custom translation template with detailed instructions"
        ))
        
        # Create a field mapping using template manager
        field_mapping = {
            "input_text": "text",
            "input_language": "source_lang",
            "output_language": "target_lang"
        }
        
        for text in texts:
            logger.info(f"Original text (English): {text}")
            
            for model_id in models_to_run:
                try:
                    result = await translate_text(
                        text=text,
                        source_lang="English",
                        target_lang="Spanish",
                        model_id=model_id,
                        api_keys=api_keys,
                        custom_template=custom_template,
                        field_mapping=field_mapping
                    )
                    
                    logger.info(f"[{model_id}] Custom template translation: {result}")
                except Exception as e:
                    logger.error(f"Error with custom template translation using {model_id}: {str(e)}")
        
        # 3. Batch translation example
        logger.info("\n--- Example 3: Batch Translation ---")
        
        batch_items = [
            {"text": text, "source_lang": "English", "target_lang": "Spanish"}
            for text in texts
        ]
        
        model_id = models_to_run[0]  # Use first model for batch example
        
        try:
            batch_results = await process_batch(
                items=batch_items,
                model_id=model_id,
                api_keys=api_keys,
                templates=default_templates
            )
            
            logger.info(f"Batch translation results with {model_id}:")
            for i, result in enumerate(batch_results):
                logger.info(f"  Text {i+1}: {result.get('response', '')}")
        except Exception as e:
            logger.error(f"Error processing batch with {model_id}: {str(e)}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
