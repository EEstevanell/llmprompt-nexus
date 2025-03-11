#!/usr/bin/env python
"""
Example script demonstrating usage of custom templates with UnifiedLLM framework.
"""
import asyncio
import os
from pathlib import Path

from llmprompt_nexus import NexusManager
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

async def demonstrate_custom_templates():
    """Demonstrate usage of both YAML-based and dictionary-based custom templates."""
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "your-key-here"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY", "your-key-here")
    }
    
    framework = NexusManager(api_keys)
    
    # Example 1: Using dictionary-based template
    logger.info("\n=== Technical Q&A Example (Dictionary-based) ===")
    qa_template = {
        "template": """Context: {context}

Question: {question}

Please provide a clear, technical answer based on the context above.""",
        "name": "technical_qa",
        "description": "Template for technical question answering",
        "system_message": "You are a technical expert. Provide accurate, technical answers based on the given context.",
        "required_variables": ["context", "question"]
    }
    
    qa_input = {
        "context": """
        The rate limiter implements a token bucket algorithm with a configurable bucket size and refill rate.
        Tokens are consumed for each API request and automatically refilled over time.
        When the bucket is empty, requests are delayed until enough tokens are available.
        """,
        "question": "How does the rate limiter handle requests when the token bucket is empty?"
    }
    result = await framework.generate(
        input_data=qa_input,
        model_id="sonar",
        template_config=qa_template  # Use template configuration directly
    )
    logger.info(f"Technical Q&A Result: {result.get('response', 'Error')}")
    
    # Example 2: Using dictionary-based template with custom system message
    logger.info("\n=== Academic Summarization Example (Dictionary-based) ===")
    summary_template = {
        "template": """Please provide a {style} summary of the following text, limited to {max_length} words:

{text}""",
        "name": "academic_summary",
        "description": "Template for academic-style summarization",
        "system_message": "You are an academic writing assistant specializing in clear, concise summaries.",
        "required_variables": ["text", "style", "max_length"]
    }
    
    summary_input = {
        "text": """
        Recent advances in transformer architectures have revolutionized natural language processing.
        The self-attention mechanism allows models to weigh different parts of the input sequence
        dynamically, enabling better handling of long-range dependencies. This has led to significant
        improvements in various NLP tasks including translation, summarization, and question answering.
        """,
        "style": "academic",
        "max_length": 50
    }
    result = await framework.generate(
        input_data=summary_input,
        model_id="sonar",
        template_config=summary_template  # Use template configuration directly
    )
    logger.info(f"Academic Summary Result: {result.get('response', 'Error')}")

if __name__ == "__main__":
    asyncio.run(demonstrate_custom_templates())