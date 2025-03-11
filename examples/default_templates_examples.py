#!/usr/bin/env python
"""
Example script demonstrating how to use core templates with the UnifiedLLM framework.
"""
import asyncio
import os
from typing import Dict, Any

from llmprompt_nexus import NexusManager
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

async def translation_example(framework: NexusManager):
    """Demonstrate translation template usage."""
    logger.info("\n=== Translation Example ===")
    input_data = {
        "text": "Hello world",
        "source_language": "English",
        "target_language": "Spanish"
    }
    result = await framework.generate(
        input_data=input_data,
        model_id="gpt-4o",
        template_name="translation"  # Will load from translation.yaml
    )
    logger.debug(f"Original: Hello world")
    logger.debug(f"Translation Result: {result.get('response', 'Error')}")

async def summarization_example(framework: NexusManager):
    """Demonstrate summarization template usage."""
    logger.info("\n=== Summarization Example ===")
    input_data = {
        "text": """
        The Python programming language was created by Guido van Rossum and was first released in 1991.
        It emphasizes code readability with its notable use of significant whitespace. Python features a
        dynamic type system and automatic memory management and supports multiple programming paradigms.
        """
    }
    result = await framework.generate(
        input_data=input_data,
        model_id="gpt-4o",
        template_name="summarization"  # Will load from summarization.yaml
    )
    logger.debug(f"Original text: {input_data['text']}")
    logger.debug(f"Summarization Result: {result.get('response', 'Error')}")

async def classification_example(framework: NexusManager):
    """Demonstrate text classification template usage."""
    logger.info("\n=== Classification Example ===")
    input_data = {
        "text": "I absolutely loved this product! Best purchase ever.",
        "categories": ["positive", "negative", "neutral"]
    }
    result = await framework.generate(
        input_data=input_data,
        model_id="gpt-4o",
        template_name="classification"  # Will load from classification.yaml
    )
    logger.debug(f"Text: {input_data['text']}")
    logger.debug(f"Classification Result: {result.get('response', 'Error')}")

async def qa_example(framework: NexusManager):
    """Demonstrate question answering template usage."""
    logger.info("\n=== Q&A Example ===")
    input_data = {
        "context": """
        The UnifiedLLM framework provides a consistent interface for working with different
        Language Model APIs. It supports template-based interactions, rate limiting,
        batch processing, and multiple providers.
        """,
        "question": "What are the main features of the UnifiedLLM framework?"
    }
    result = await framework.generate(
        input_data=input_data,
        model_id="gpt-4o",
        template_name="qa"  # Will load from qa.yaml
    )
    logger.debug(f"Context: {input_data['context']}")
    logger.debug(f"Question: {input_data['question']}")
    logger.debug(f"Q&A Result: {result.get('response', 'Error')}")

async def intent_example(framework: NexusManager):
    """Demonstrate intent detection template usage."""
    logger.info("\n=== Intent Detection Example ===")
    input_data = {
        "text": "Can you help me book a flight to New York?",
        "possible_intents": ["booking", "information", "support", "other"]
    }
    result = await framework.generate(
        input_data=input_data,
        model_id="gpt-4o",
        template_name="intent"  # Will load from intent.yaml
    )
    logger.debug(f"Text: {input_data['text']}")
    logger.debug(f"Intent Result: {result.get('response', 'Error')}")

async def main():
    """Run template usage examples."""
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "your-key-here"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY", "your-key-here")
    }
    
    framework = NexusManager(api_keys)
    
    await translation_example(framework)
    await summarization_example(framework)
    await classification_example(framework)
    await qa_example(framework)
    await intent_example(framework)

if __name__ == "__main__":
    asyncio.run(main())