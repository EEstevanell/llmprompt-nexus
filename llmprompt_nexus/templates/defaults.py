"""
Default template configuration for UnifiedLLM framework.
"""
from pathlib import Path
from typing import Dict, Any, Optional

from llmprompt_nexus.templates.manager import TemplateManager

def get_template_manager(template_type: str = 'translation') -> TemplateManager:
    """
    Get template manager for specified template type by loading directly from config file.
    
    Args:
        template_type: Type of templates to load (e.g., 'translation', 'qa')
        
    Returns:
        TemplateManager with templates from the specified type
    """
    config_dir = Path(__file__).parent / 'config' / 'templates'
    type_file = config_dir / f"{template_type}.yaml"
    
    if not type_file.exists():
        raise ValueError(f"Template configuration file not found: {type_file}")
        
    return TemplateManager.from_yaml(type_file)

def render_template(template_name: str, variables: Dict[str, Any], template_type: str = 'translation') -> str:
    """
    Render a template by name using the specified variables.
    Templates are loaded directly from config files.
    
    Args:
        template_name: Name of the template to render
        variables: Dictionary of variables to use in rendering
        template_type: Type of template to load
        
    Returns:
        Rendered template string
    """
    manager = get_template_manager(template_type)
    template = manager.get_template(template_name)
    return template.render(variables)