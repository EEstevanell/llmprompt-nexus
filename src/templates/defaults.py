"""
Default template configuration for UnifiedLLM framework.
"""
from typing import Dict, Any

from src.templates.registry import registry
from src.templates.manager import TemplateManager
from src.templates.base import TemplateStrategy

# Core template types supported by the framework
TEMPLATE_TYPES = [
    'sentiment',
    'classification',
    'intent',
    'qa',
    'summarization',
    'translation'
]

def get_template_manager(template_type: str = 'translation') -> TemplateManager:
    """
    Get a template manager for a specific template type.
    
    Args:
        template_type: One of the supported template types
        
    Returns:
        TemplateManager instance for the specified type
        
    Raises:
        ValueError: If template type is not supported
    """
    if template_type not in TEMPLATE_TYPES:
        raise ValueError(f"Unsupported template type: {template_type}. Must be one of: {TEMPLATE_TYPES}")
        
    domain_manager = registry.get_domain(template_type)
    return domain_manager

def render_template(template_name: str, variables: Dict[str, Any], template_type: str = 'translation') -> str:
    """
    Render a template with the given variables.
    
    Args:
        template_name: Name of the template to render
        variables: Dictionary of variables for template rendering
        template_type: Type of template to use
        
    Returns:
        Rendered template string
    """
    manager = get_template_manager(template_type)
    return manager.render_template(template_name, variables)