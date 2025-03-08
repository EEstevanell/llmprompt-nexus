#!/usr/bin/env python
"""
Example script demonstrating how to use core templates with the UnifiedLLM framework.
"""
import asyncio
import os
from typing import Dict, Any

from src.core.framework import UnifiedLLM
from src.templates.defaults import get_template_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def translation_example(framework: UnifiedLLM):
    """Demonstrate translation template usage."""
    logger.info("\n=== Translation Template Example ===")
    
    tm = get_template_manager('translation')
    
    text = "The autumn leaves danced in the crisp morning breeze."
    input_data = {
        "text": text,
        "source_language": "English",
        "target_language": "Spanish"
    }
    
    result = await framework.run_with_model(
        input_data=input_data,
        model_id="sonar-pro",  # This is now the model name in the config
        template=tm.get_template('translation')
    )
    
    logger.info(f"Original: {text}")
    logger.info(f"Translation: {result.get('response', 'Error')}")

async def summarization_example(framework: UnifiedLLM):
    """Demonstrate summarization template usage."""
    logger.info("\n=== Summarization Template Example ===")
    
    tm = get_template_manager('summarization')
    
    text = """
    Recent studies in machine learning have shown significant improvements in natural language processing tasks. 
    Through extensive experimentation with transformer architectures, researchers have demonstrated that 
    pre-trained language models can achieve state-of-the-art results across multiple benchmarks. 
    The methodology involved training on large-scale datasets and fine-tuning for specific tasks. 
    Results indicate a 15% improvement over baseline models, suggesting that this approach could 
    revolutionize how we approach NLP tasks in production environments.
    """
    
    input_data = {
        "text": text,
        "length": "brief"
    }
    
    result = await framework.run_with_model(
        input_data=input_data,
        model_id="sonar-pro",
        template=tm.get_template('summarization')
    )
    
    logger.info(f"Original length: {len(text)} characters")
    logger.info(f"Summary: {result.get('response', 'Error')}")

async def classification_example(framework: UnifiedLLM):
    """Demonstrate text classification template usage."""
    logger.info("\n=== Text Classification Template Example ===")
    
    tm = get_template_manager('classification')
    
    text = """
    To install the package, run 'pip install unifiedllm' in your terminal. 
    Make sure you have Python 3.8 or higher installed on your system.
    """
    
    input_data = {
        "text": text,
        "categories": ["Documentation", "Tutorial", "Error Message", "Code Snippet"]
    }
    
    result = await framework.run_with_model(
        input_data=input_data,
        model_id="sonar-pro",
        template=tm.get_template('classification')
    )
    
    logger.info(f"Text: {text}")
    logger.info(f"Classification: {result.get('response', 'Error')}")

async def qa_example(framework: UnifiedLLM):
    """Demonstrate question answering template usage."""
    logger.info("\n=== Question Answering Template Example ===")
    
    tm = get_template_manager('qa')
    
    context = """
    The UnifiedLLM framework provides a standardized interface for working with different 
    language models. It supports multiple template types including translation, summarization, 
    sentiment analysis, classification, and question answering. The framework handles API rate 
    limiting and provides consistent error handling across different model providers.
    """
    
    input_data = {
        "context": context,
        "question": "What template types does the UnifiedLLM framework support?"
    }
    
    result = await framework.run_with_model(
        input_data=input_data,
        model_id="sonar-pro",
        template=tm.get_template('qa')
    )
    
    logger.info(f"Question: {input_data['question']}")
    logger.info(f"Answer: {result.get('response', 'Error')}")

async def intent_example(framework: UnifiedLLM):
    """Demonstrate intent detection template usage."""
    logger.info("\n=== Intent Detection Template Example ===")
    
    tm = get_template_manager('intent')
    
    text = "Can you help me translate this document from English to Spanish?"
    
    input_data = {
        "text": text
    }
    
    result = await framework.run_with_model(
        input_data=input_data,
        model_id="sonar-pro",
        template=tm.get_template('intent')
    )
    
    logger.info(f"Text: {text}")
    logger.info(f"Intent Analysis: {result.get('response', 'Error')}")

async def main():
    """Run template usage examples."""
    try:
        # Initialize framework with API keys
        api_keys = {
            "perplexity": os.getenv("PERPLEXITY_API_KEY")  or "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",
            "openai": os.getenv("OPENAI_API_KEY")
        }
        
        framework = UnifiedLLM(api_keys)
        
        # Run examples for each core template type
        await translation_example(framework)
        await summarization_example(framework)
        await classification_example(framework)
        await qa_example(framework)
        await intent_example(framework)
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())