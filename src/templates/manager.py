"""
Template management system for UnifiedLLM framework.
"""
from typing import Dict, Any, Optional, List, Set, Type, Union
import os
import yaml
from pathlib import Path
import json

from src.templates.base import Template, TemplateStrategy, IntentionTemplateStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TemplateManager:
    """
    Manager for template collections and registry.
    
    This class handles loading, registering, and retrieving templates,
    as well as applying strategies to select and render templates.
    """
    
    def __init__(self, templates: Optional[Dict[str, Template]] = None):
        """
        Initialize TemplateManager with optional initial templates.
        
        Args:
            templates: Optional dictionary of template name -> Template object
        """
        self.templates: Dict[str, Template] = templates or {}
        self.default_strategy: TemplateStrategy = IntentionTemplateStrategy()
    
    def register_template(self, template: Template) -> None:
        """
        Register a template with the manager.
        
        Args:
            template: The template object to register
        """
        logger.debug(f"Registering template: {template.name}")
        self.templates[template.name] = template
    
    def register_templates(self, templates: Dict[str, Template]) -> None:
        """
        Register multiple templates at once.
        
        Args:
            templates: Dictionary of template name -> Template object
        """
        for name, template in templates.items():
            self.register_template(template)
    
    def get_template(self, name: str) -> Template:
        """
        Get a template by name.
        
        Args:
            name: Name of the template to retrieve
            
        Returns:
            Template object
            
        Raises:
            ValueError: If the template is not found
        """
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        return self.templates[name]
    
    def list_templates(self) -> List[str]:
        """
        List all registered template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def render_template(self, name: str, variables: Dict[str, Any], include_system: bool = True) -> Union[str, List[Dict[str, str]]]:
        """
        Render a template by name with the provided variables.
        
        Args:
            name: Name of the template to render
            variables: Dictionary of variable values
            include_system: If True, returns messages list with system message if present
            
        Returns:
            Either rendered template text or list of messages if include_system=True
        """
        template = self.get_template(name)
        if include_system and template.system_message is not None:
            return template.get_messages(variables)
        return template.render(variables)
    
    def apply_template(self, input_data: Dict[str, Any], 
                     strategy: Optional[TemplateStrategy] = None,
                     include_system: bool = True) -> Union[str, List[Dict[str, str]]]:
        """
        Apply a template using a strategy.
        
        This method selects a template and renders it using the provided
        strategy (or the default strategy).
        
        Args:
            input_data: Input data containing intention/task and variables
            strategy: Strategy to use for template selection and variable mapping
            include_system: If True, returns messages list with system message if present
            
        Returns:
            Either rendered template text or list of messages if include_system=True
        """
        strategy = strategy or self.default_strategy
        
        # Select appropriate template
        template = strategy.select_template(input_data, self.templates)
        
        # Prepare variables
        variables = strategy.prepare_variables(input_data, template)
        
        # Render the template
        if include_system and template.system_message is not None:
            return template.get_messages(variables)
        return template.render(variables)
    
    def load_from_dict(self, template_dict: Dict[str, Any]) -> None:
        """
        Load templates from a dictionary that may include system messages.
        
        Args:
            template_dict: Dictionary with template configurations
        """
        for name, config in template_dict.items():
            if isinstance(config, str):
                # Simple string template
                self.register_template(Template(
                    template_text=config,
                    name=name
                ))
            elif isinstance(config, dict):
                # Template with metadata
                self.register_template(Template(
                    template_text=config['template'],
                    name=name,
                    description=config.get('description', ''),
                    system_message=config.get('system_message')
                ))
            else:
                logger.warning(f"Skipping invalid template config for {name}")
    
    def load_from_yaml(self, file_path: str) -> None:
        """
        Load templates from a YAML file.
        
        Expected YAML format:
        ```yaml
        templates:
          translation:
            template: "Translate this text: {text}"
            description: "Basic translation template"
            system_message: "You are an expert translator."
          summarize:
            template: "Summarize this text: {text}"
            system_message: "You are an expert at summarizing content."
        ```
        
        Args:
            file_path: Path to the YAML file
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'templates' not in data:
                raise ValueError(f"YAML file {file_path} does not contain 'templates' section")
            
            for name, template_data in data['templates'].items():
                if isinstance(template_data, str):
                    template_text = template_data
                    system_message = None
                    description = ''
                else:
                    template_text = template_data.get('template')
                    if not template_text:
                        logger.warning(f"Skipping template '{name}' without template text")
                        continue
                    system_message = template_data.get('system_message')
                    description = template_data.get('description', '')
                
                self.register_template(Template(
                    template_text=template_text,
                    name=name,
                    description=description,
                    system_message=system_message
                ))
                
            logger.info(f"Loaded {len(data['templates'])} templates from {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading templates from {file_path}: {str(e)}")
            raise
    
    def save_to_yaml(self, file_path: str) -> None:
        """
        Save all templates to a YAML file.
        
        Args:
            file_path: Path to save the YAML file
        """
        try:
            data = {'templates': {}}
            
            for name, template in self.templates.items():
                template_data = {
                    'template': template.template_text,
                    'description': template.description
                }
                if template.system_message:
                    template_data['system_message'] = template.system_message
                    
                data['templates'][name] = template_data
            
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
            logger.info(f"Saved {len(self.templates)} templates to {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving templates to {file_path}: {str(e)}")
            raise
    
    @classmethod
    def from_legacy_dict(cls, legacy_templates: Dict[str, str]) -> 'TemplateManager':
        """
        Create a TemplateManager from a legacy template dictionary.
        
        Args:
            legacy_templates: Dictionary mapping template name to template text
            
        Returns:
            New TemplateManager instance with the converted templates
        """
        manager = cls()
        manager.load_from_dict(legacy_templates)
        return manager


# Default instance for common use
template_manager = TemplateManager()