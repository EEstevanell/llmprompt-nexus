#!/usr/bin/env python
"""
Example script demonstrating how to create and use custom templates with system messages.
"""
import asyncio
import os
from typing import Dict

from src.core.framework import UnifiedLLM
from src.templates.manager import TemplateManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Example custom templates with system messages
custom_templates = {
    "expert_translation": {
        "template": "Translate the following text from {source_language} to {target_language}:\n\n{text}",
        "description": "Professional translation template with expert system context",
        "system_message": "You are an expert translator with deep knowledge of both source and target languages, including cultural nuances and context."
    },
    "technical_qa": {
        "template": "Based on the following technical documentation, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}",
        "description": "Technical documentation Q&A template",
        "system_message": "You are a technical expert who provides clear, accurate, and concise answers based on documentation. Focus on technical accuracy and practical implementation details."
    },
    "academic_summarization": {
        "template": "Provide a {length} summary of the following academic text:\n\n{text}",
        "description": "Academic text summarization template",
        "system_message": "You are an academic researcher skilled in distilling complex academic content into clear summaries while preserving key technical details and maintaining academic rigor."
    }
}

async def demonstrate_custom_templates():
    """Demonstrate usage of custom templates with system messages."""
    # Initialize framework with API keys
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY") or "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",
        "openai": os.getenv("OPENAI_API_KEY")
    }
    
    framework = UnifiedLLM(api_keys)
    
    # Create a template manager and load custom templates
    tm = TemplateManager()
    tm.load_from_dict(custom_templates)
    
    # Example 1: Expert Translation
    logger.info("\n=== Expert Translation Example ===")
    translation_input = {
        "text": "The implementation requires careful consideration of edge cases and performance implications.",
        "source_language": "English",
        "target_language": "Spanish"
    }
    result = await framework.run_with_model(
        input_data=translation_input,
        model_id="sonar-pro",
        template=tm.get_template('expert_translation')
    )
    logger.info(f"Expert Translation Result: {result.get('response', 'Error')}")
    
    # Example 2: Technical Q&A
    logger.info("\n=== Technical Q&A Example ===")
    qa_input = {
        "context": """
        The rate limiter implements a token bucket algorithm with a configurable bucket size and refill rate.
        Tokens are consumed for each API request and automatically refilled over time.
        When the bucket is empty, requests are delayed until enough tokens are available.
        """,
        "question": "How does the rate limiter handle requests when the token bucket is empty?"
    }
    result = await framework.run_with_model(
        input_data=qa_input,
        model_id="sonar-pro",
        template=tm.get_template('technical_qa')
    )
    logger.info(f"Technical Q&A Result: {result.get('response', 'Error')}")
    
    # Example 3: Academic Summarization
    logger.info("\n=== Academic Summarization Example ===")
    summary_input = {
        "text": """
        Recent advances in transformer-based language models have revolutionized natural language processing tasks.
        Through the implementation of multi-head self-attention mechanisms, these models can capture long-range
        dependencies and contextual relationships in text data more effectively than traditional recurrent neural
        networks. The architecture's parallel processing capabilities also enable significant improvements in
        training efficiency and scalability.
        """,
        "length": "concise"
    }
    result = await framework.run_with_model(
        input_data=summary_input,
        model_id="sonar-pro",
        template=tm.get_template('academic_summarization')
    )
    logger.info(f"Academic Summary Result: {result.get('response', 'Error')}")

if __name__ == "__main__":
    asyncio.run(demonstrate_custom_templates())