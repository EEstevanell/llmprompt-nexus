"""
Template management system for UnifiedLLM framework.
"""
from typing import Dict, Any, Optional, List, Set, Union
import os
import yaml
from pathlib import Path

from src.templates.base import Template
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TemplateManager:
    """
    Manager for template collections and registry.
    
    This class handles loading, registering, and retrieving templates from YAML files.
    Templates are configured in config/templates/ with each task type having its own file.
    """
    
    def __init__(self, templates: Optional[Dict[str, Template]] = None):
        """Initialize with optional predefined templates."""
        self.templates = templates or {}
    
    def register_template(self, template: Template) -> None:
        """Register a new template."""
        self.templates[template.name] = template
    
    def register_templates(self, templates: Dict[str, Template]) -> None:
        """Register multiple templates at once."""
        self.templates.update(templates)
    
    def get_template(self, name: str) -> Template:
        """Get a template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name]
    
    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self.templates.keys())
    
    def render_template(self, name: str, variables: Dict[str, Any], include_system: bool = True) -> Union[str, List[Dict[str, str]]]:
        """
        Render a template by name.
        
        Args:
            name: Name of template to render
            variables: Variables to use in template
            include_system: If True, returns messages list with system message if present
            
        Returns:
            Either rendered template text or list of messages if include_system=True
        """
        template = self.get_template(name)
        
        # Prepare and validate variables
        variables = template.prepare_variables(variables)
        
        # Return appropriate format
        if include_system and template.system_message is not None:
            return template.get_messages(variables)
        return template.render(variables)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'TemplateManager':
        """
        Create a TemplateManager from a YAML file.
        
        The YAML file should have this structure:
        ```yaml
        templates:
          template_name:
            template: "Template text with {variables}"
            description: "Template description"
            system_message: "Optional system message"
        ```
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            
        if not isinstance(data, dict) or 'templates' not in data:
            raise ValueError("Invalid template YAML file format")
            
        templates = {}
        for name, config in data['templates'].items():
            templates[name] = Template(
                template_text=config['template'],
                name=name,
                description=config.get('description', ''),
                system_message=config.get('system_message')
            )
            
        return cls(templates)
    
    @classmethod
    def from_yaml_dir(cls, dir_path: Union[str, Path]) -> 'TemplateManager':
        """Create a TemplateManager from a directory of YAML files."""
        manager = cls()
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            logger.warning(f"Template directory {dir_path} does not exist")
            return manager
            
        for yaml_file in dir_path.glob('*.yaml'):
            try:
                other_manager = cls.from_yaml(yaml_file)
                manager.register_templates(other_manager.templates)
            except Exception as e:
                logger.error(f"Error loading templates from {yaml_file}: {str(e)}")
                
        return manager

# Default instance for common use
template_manager = TemplateManager()