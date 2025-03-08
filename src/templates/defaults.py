# src/templates/translation.py
"""
Translation templates for the UnifiedLLM framework.
"""
from typing import Dict

from src.templates.base import Template, TranslationTemplateStrategy
from src.templates.manager import TemplateManager

# Create standard translation templates
translation_template = Template(
    template_text="""
    Please translate the following text:
    
    Original text ({source_lang}):
    {text}
    
    Translate to {target_lang} while maintaining the original meaning, tone, and style.
    Please ensure proper grammar and natural expression in the target language.
    
    Guidelines:
    - Maintain any technical terminology accurately
    - Preserve formatting and structure
    - Keep named entities unchanged unless there's a widely accepted translation
    - Handle idiomatic expressions appropriately for the target culture
    """,
    name="translate",
    description="Standard template for text translation between languages"
)

default_template = Template(
    template_text="""
    Please answer the following question:
    {question}
    """,
    name="default",
    description="Default template for general queries"
)

summarize_template = Template(
    template_text="""
    Please summarize the following text:
    {text}
    """,
    name="summarize",
    description="Template for text summarization"
)

sentiment_template = Template(
    template_text="""
    Analyze the sentiment of the following text:
    {text}
    
    Please classify it as positive, negative, or neutral and explain why.
    """,
    name="sentiment",
    description="Template for sentiment analysis of text"
)

# Create a template manager specifically for translation tasks
default_manager = TemplateManager()
default_manager.register_template(translation_template)
default_manager.register_template(default_template)
default_manager.register_template(summarize_template)
default_manager.register_template(sentiment_template)

# Set the default strategy to TranslationTemplateStrategy
default_manager.default_strategy = TranslationTemplateStrategy()

# Legacy dictionary for backward compatibility
templates = {
    "default": default_template.template_text,
    "summarize": summarize_template.template_text,
    "sentiment": sentiment_template.template_text,
    "translate": translation_template.template_text
}

def get_template_manager() -> TemplateManager:
    """
    Get the translation template manager with predefined translation templates.
    
    Returns:
        TemplateManager instance with translation templates
    """
    return default_manager

def render_template(template_name: str, variables: Dict) -> str:
    """
    Render a translation template with the provided variables.
    
    Args:
        template_name: Name of the template to render
        variables: Dictionary of variables for the template
    
    Returns:
        Rendered template text
    """
    return default_manager.render_template(template_name, variables)