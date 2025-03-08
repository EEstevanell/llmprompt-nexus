"""
Default template configuration for UnifiedLLM framework.
"""
from pathlib import Path
from typing import Dict, Any

from src.templates.registry import registry
from src.templates.manager import TemplateManager

# Core template types supported by the framework
TEMPLATE_TYPES = [
    'classification',
    'intent',
    'qa',
    'summarization',
    'translation'
]

def get_template_manager(template_type: str = 'translation') -> TemplateManager:
    """
    Get template manager for specified template type.
    
    Args:
        template_type: Type of templates to load (e.g., 'translation', 'qa')
        
    Returns:
        TemplateManager with templates loaded from config/templates/{template_type}.yaml
    """
    if template_type not in TEMPLATE_TYPES:
        raise ValueError(f"Unknown template type: {template_type}")
        
    config_dir = Path(__file__).parent.parent.parent / 'config' / 'templates'
    config_file = config_dir / f"{template_type}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Template config file not found: {config_file}")
        
    return TemplateManager.from_yaml(config_file)

def render_template(template_name: str, variables: Dict[str, Any], template_type: str = 'translation') -> str:
    """
    Helper function to render a template by name.
    
    Args:
        template_name: Name of template to render
        variables: Variables to use in template
        template_type: Type of template to use (e.g., 'translation', 'qa')
        
    Returns:
        Rendered template text
    """
    manager = get_template_manager(template_type)
    template = manager.get_template(template_name)
    return template.render(variables)