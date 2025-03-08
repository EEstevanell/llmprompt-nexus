"""
Template management system for UnifiedLLM framework.
"""
from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path

from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.utils.logger import get_logger

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
    def from_yaml(cls, file_path: Path) -> 'TemplateManager':
        """
        Create a TemplateManager instance by loading templates from a YAML file.
        
        Args:
            file_path: Path to the YAML file containing template definitions
            
        Returns:
            TemplateManager instance with loaded templates
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config or 'templates' not in config:
            raise ValueError(f"Invalid template configuration in {file_path}")
            
        templates = {}
        for name, template_config in config['templates'].items():
            template = Template(
                template_text=template_config['template'],
                name=name,
                description=template_config.get('description', ''),
                system_message=template_config.get('system_message')
            )
            templates[name] = template
            
        return cls(templates)

    @classmethod
    def from_yaml_dir(cls, dir_path: Path) -> 'TemplateManager':
        """
        Create a TemplateManager instance by loading all YAML files in a directory.
        
        Args:
            dir_path: Path to directory containing template YAML files
            
        Returns:
            TemplateManager instance with all templates loaded from the directory
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"Template directory not found: {dir_path}")
            
        templates = {}
        for file_path in dir_path.glob('*.yaml'):
            try:
                manager = cls.from_yaml(file_path)
                templates.update(manager.templates)
            except Exception as e:
                logger.warning(f"Error loading templates from {file_path}: {str(e)}")
                continue
                
        return cls(templates)

# Default instance for common use
template_manager = TemplateManager()