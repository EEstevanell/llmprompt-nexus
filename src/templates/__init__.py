"""
Template system for UnifiedLLM framework.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union

from src.templates.base import Template
from src.templates.manager import TemplateManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_template(template_name: str = None, template_config: Dict[str, Any] = None) -> Template:
    """
    Load a template by name from YAML file or from config dictionary.
    
    Args:
        template_name: Optional name of the template to load from YAML file
        template_config: Optional template configuration dictionary
        
    Returns:
        Template instance
        
    Raises:
        ValueError: If template cannot be found or is invalid
    """
    # If config is provided directly, validate and create template
    if template_config:
        Template.validate_template_config(template_config)
        return Template(
            template_text=template_config['template'],
            name=template_config.get('name', 'custom_template'),
            description=template_config.get('description', ''),
            system_message=template_config.get('system_message'),
            required_variables=set(template_config.get('required_variables', []))
        )
    
    if not template_name:
        raise ValueError("Either template_name or template_config must be provided")
    
    # Look for template in config files
    config_dir = Path(__file__).parent.parent.parent / 'config' / 'templates'
    yaml_file = config_dir / f"{template_name}.yaml"
    
    if not yaml_file.exists():
        raise ValueError(f"Template file not found: {yaml_file}")
    
    try:
        manager = TemplateManager.from_yaml(yaml_file)
        template_key = next(iter(manager.templates.keys()))  # Get first template from file
        return manager.get_template(template_key)
    except Exception as e:
        logger.error(f"Error loading template from {yaml_file}: {str(e)}")
        raise ValueError(f"Failed to load template '{template_name}': {str(e)}")

def render_template(template_name: str = None, 
                   variables: Dict[str, Any] = None,
                   template_config: Dict[str, Any] = None) -> Union[str, Dict[str, str]]:
    """
    Render a template by name using the specified variables.
    
    Args:
        template_name: Optional name of the template YAML file to load
        variables: Dictionary of variables to use in rendering
        template_config: Optional template configuration dictionary
        
    Returns:
        Rendered template string or messages dictionary if template has system message
    """
    template = load_template(template_name, template_config)
    if template.system_message:
        return template.get_messages(variables)
    return template.render(variables)