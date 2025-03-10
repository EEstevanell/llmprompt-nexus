#!/usr/bin/env python
"""
Example script demonstrating batch processing capabilities of the llmprompt-nexus framework.
This example showcases the automatic queue-based batching system that respects rate limits.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Any

from llmprompt_nexus import NexusManager
from llmprompt_nexus.models.registry import registry as model_registry
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

# Example custom template with system message
batch_translation_template = {
    "template": "Translate the following text from {source_language} to {target_language}:\n\n{text}",
    "description": "Optimized translation template for batch processing",
    "system_message": "You are an expert translator specializing in batch translation tasks. Focus on maintaining consistency across all translations while preserving meaning and context."
}

async def demonstrate_basic_batch():
    """Demonstrate basic batching with automatic queue management."""
    logger.info("\n=== Basic Batch Processing with Auto Queue Management ===")
    
    # Initialize framework with API keys
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "dummy-key-for-example")
    }
    
    framework = NexusManager(api_keys)
    
    # Prepare batch of prompts
    prompts = [
        "Write a short limerick about programming.",
        "Write a haiku about artificial intelligence.",
        "Create a brief sonnet about data science.",
        "Write a short poem about machine learning.",
        "Create a rhyming couplet about algorithms.",
        "Write a brief stanza about neural networks.",
        "Create a quatrain about software engineering.",
        "Write a brief verse about cloud computing.",
        "Create a short rhyme about databases.",
        "Write a brief poem about cybersecurity."
    ]
    
    # Display information about model rate limiting
    model_id = "gpt-3.5-turbo"
    model_config = model_registry.get_model(model_id)
    client = framework.get_client(model_config.provider)
    rate_limiter = client.get_rate_limiter(model_config.name)
    usage = rate_limiter.get_current_usage()
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Rate limit: {usage['max_calls']} calls per {usage['period']} seconds")
    logger.info(f"Current usage: {usage['calls']} calls")
    logger.info(f"Batch size: {len(prompts)} prompts")
    
    # Process batch with auto queue management
    start_time = time.time()
    logger.info("\nStarting batch processing...")
    
    # Using the new simplified API - we can pass string prompts directly
    # The default template will be used automatically
    results = await framework.process_batch(
        inputs=prompts,
        model_id=model_id
    )
    
    end_time = time.time()
    
    # Display results
    logger.info(f"\nBatch completed in {end_time - start_time:.2f} seconds")
    logger.info("\nResults:")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        logger.info(f"Result: {result.get('response', 'Error')}")

async def demonstrate_template_batch():
    """Demonstrate batch processing using templates with automatic queue management."""
    logger.info("\n=== Template-based Batch Processing with Auto Queue Management ===")
    
    # Initialize framework
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "dummy-key-for-example")
    }
    
    framework = NexusManager(api_keys)
    
    # Prepare texts for batch translation
    texts = [
        "The API implements robust rate limiting mechanisms.",
        "Data structures are optimized for concurrent access.",
        "The framework supports asynchronous batch processing.",
        "Unit tests ensure code reliability.",
        "Configuration is managed through YAML files.",
        "Automated testing improves code quality.",
        "Documentation is essential for developer onboarding.",
        "Version control systems track code changes.",
        "CI/CD pipelines automate deployment processes.",
        "Code reviews help identify potential issues."
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
    
    # Display information about the batch
    model_id = "gpt-3.5-turbo"
    model_config = model_registry.get_model(model_id)
    client = framework.get_client(model_config.provider)
    rate_limiter = client.get_rate_limiter(model_config.name)
    usage = rate_limiter.get_current_usage()
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Rate limit: {usage['max_calls']} calls per {usage['period']} seconds")
    logger.info(f"Current usage: {usage['calls']} calls")
    logger.info(f"Batch size: {len(batch_inputs)} items")
    
    # Process batch with template
    start_time = time.time()
    logger.info("\nStarting batch processing with template...")
    
    # Using the new simplified API - we can provide the custom template configuration directly
    results = await framework.process_batch(
        inputs=batch_inputs,
        model_id=model_id,
        template_config=batch_translation_template
    )
    
    end_time = time.time()
    
    # Display results
    logger.info(f"\nBatch completed in {end_time - start_time:.2f} seconds")
    logger.info("\nTranslation results:")
    for i, (text, result) in enumerate(zip(texts, results)):
        logger.info(f"\nOriginal ({i+1}): {text}")
        logger.info(f"Translation: {result.get('response', 'Error')}")

async def demonstrate_file_processing():
    """Demonstrate file processing with automatic batching."""
    logger.info("\n=== File Processing with Auto Batching ===")
    
    # Initialize framework
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "dummy-key-for-example")
    }
    
    framework = NexusManager(api_keys)
    
    # Create a test TSV file if it doesn't exist
    file_path = Path("./test_batch.tsv")
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write("id\ttext\n")
            for i in range(1, 11):
                f.write(f"{i}\tThis is test sentence {i} that needs processing.\n")
        logger.info(f"Created test file: {file_path}")
    
    # Get model information
    model_id = "gpt-3.5-turbo"
    
    # Create template configuration
    template_config = {
        "template": "Analyze the sentiment of the following text and classify it as positive, negative, or neutral:\n\n{text}",
        "system_message": "You are an expert at sentiment analysis. Provide only the sentiment label without explanation."
    }
    
    # Display information about the batch processing
    logger.info(f"Processing file: {file_path}")
    logger.info(f"Model: {model_id}")
    client = framework.get_client(model_registry.get_model(model_id).provider)
    rate_limiter = client.get_rate_limiter(model_registry.get_model(model_id).name)
    usage = rate_limiter.get_current_usage()
    logger.info(f"Rate limit: {usage['max_calls']} calls per {usage['period']} seconds")
    
    # Process file with auto batching
    start_time = time.time()
    logger.info("\nStarting file processing...")
    
    try:
        # Using the new simplified API with consistent parameter naming
        output_path = await framework.process_file(
            file_path=file_path,
            model_id=model_id,
            template_config=template_config
        )
        
        end_time = time.time()
        logger.info(f"\nFile processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

async def demonstrate_single_processing():
    """Demonstrate processing of a single request."""
    logger.info("\n=== Single Request Processing ===")
    
    # Initialize framework
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "dummy-key-for-example")
    }
    
    framework = NexusManager(api_keys)
    model_id = "gpt-3.5-turbo"
    
    # Example 1: Simple string prompt (will use default template)
    logger.info("\nExample 1: Simple string prompt")
    simple_prompt = "Explain the concept of neural networks in 3 sentences."
    
    start_time = time.time()
    result1 = await framework.process(
        input_data=simple_prompt,
        model_id=model_id
    )
    end_time = time.time()
    
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Prompt: {simple_prompt}")
    logger.info(f"Response: {result1.get('response', 'Error')}")
    
    # Example 2: Dictionary with custom template
    logger.info("\nExample 2: Dictionary with custom template")
    template_config = {
        "template": "Provide a brief {length} explanation of {concept} for someone with {background} knowledge.",
        "system_message": "You are an educational AI that specializes in providing concise, targeted explanations."
    }
    
    start_time = time.time()
    result2 = await framework.process(
        input_data={
            "concept": "quantum computing", 
            "length": "3-sentence", 
            "background": "beginner"
        },
        model_id=model_id,
        template_config=template_config
    )
    end_time = time.time()
    
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info("Input: concept=quantum computing, length=3-sentence, background=beginner")
    logger.info(f"Response: {result2.get('response', 'Error')}")

async def main():
    """Run batch processing examples."""
    try:
        await demonstrate_single_processing()
        await demonstrate_basic_batch()
        await demonstrate_template_batch()
        await demonstrate_file_processing()
    except Exception as e:
        logger.error(f"Error running batch examples: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())