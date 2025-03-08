"""
Central template registry system for the UnifiedLLM framework.

The template registry provides a centralized location to register,
discover and manage templates across the framework.
"""
from typing import Dict, List, Optional, Any, Set
import os
import glob
import yaml
from pathlib import Path

from src.templates.base import Template, TemplateStrategy
from src.templates.manager import TemplateManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TemplateRegistry:
    """
    Central registry for templates across the application.
    
    This class provides a unified interface for accessing templates
    from different domains (translation, intention, etc.), and makes
    it easier to discover and extend templates.
    """
    
    def __init__(self):
        """Initialize the template registry with empty managers."""
        # Dictionary mapping domain names to TemplateManager instances
        self.domain_managers: Dict[str, TemplateManager] = {}
        
        # Default templates directory
        self.templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config", "templates")
        
    def register_domain(self, domain: str, manager: TemplateManager) -> None:
        """
        Register a template manager for a specific domain.
        
        Args:
            domain: The domain name (e.g., 'translation', 'intention')
            manager: The template manager for this domain
        """
        logger.info(f"Registering template domain: {domain}")
        self.domain_managers[domain] = manager
        
    def get_domain(self, domain: str) -> TemplateManager:
        """
        Get the template manager for a specific domain.
        
        Args:
            domain: The domain name
            
        Returns:
            The template manager for the domain
            
        Raises:
            ValueError: If the domain is not registered
        """
        if domain not in self.domain_managers:
            raise ValueError(f"Template domain not found: {domain}")
        return self.domain_managers[domain]
    
    def list_domains(self) -> List[str]:
        """
        List all registered template domains.
        
        Returns:
            List of domain names
        """
        return list(self.domain_managers.keys())
    
    def get_template(self, domain: str, template_name: str) -> Template:
        """
        Get a template from a specific domain.
        
        Args:
            domain: The domain name
            template_name: The template name within the domain
            
        Returns:
            The template object
            
        Raises:
            ValueError: If the domain or template is not found
        """
        manager = self.get_domain(domain)
        return manager.get_template(template_name)
    
    def list_templates(self, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all templates, optionally filtered by domain.
        
        Args:
            domain: Optional domain to filter by
            
        Returns:
            Dictionary mapping domain names to lists of template names
        """
        if domain:
            return {domain: self.get_domain(domain).list_templates()}
        
        return {
            domain: manager.list_templates()
            for domain, manager in self.domain_managers.items()
        }
    
    def render_template(self, domain: str, template_name: str, variables: Dict[str, Any]) -> str:
        """
        Render a template from a specific domain.
        
        Args:
            domain: The domain name
            template_name: The template name within the domain
            variables: Dictionary of variables for template rendering
            
        Returns:
            Rendered template text
            
        Raises:
            ValueError: If the domain or template is not found or variables are missing
        """
        manager = self.get_domain(domain)
        return manager.render_template(template_name, variables)
    
    def apply_template(self, domain: str, input_data: Dict[str, Any], 
                    strategy: Optional[TemplateStrategy] = None) -> str:
        """
        Apply a template from a specific domain using a strategy.
        
        Args:
            domain: The domain name
            input_data: Input data for template selection and rendering
            strategy: Optional strategy to use (defaults to domain's default strategy)
            
        Returns:
            Rendered template text
            
        Raises:
            ValueError: If the domain is not found or template cannot be applied
        """
        manager = self.get_domain(domain)
        return manager.apply_template(input_data, strategy)
    
    def load_templates_from_directory(self, directory: Optional[str] = None) -> None:
        """
        Load templates from YAML files in a directory.
        
        This method scans a directory for YAML files and loads them as template domains.
        The filename (without extension) is used as the domain name.
        
        Args:
            directory: Directory path to scan for template files
                      (defaults to config/templates in the project root)
        """
        directory = directory or self.templates_dir
        
        try:
            if not os.path.exists(directory):
                logger.warning(f"Templates directory does not exist: {directory}")
                return
            
            yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
            if not yaml_files:
                logger.warning(f"No template YAML files found in {directory}")
                return
                
            for yaml_file in yaml_files:
                try:
                    # Use filename as domain name
                    domain = os.path.splitext(os.path.basename(yaml_file))[0]
                    
                    # Create a new manager for this domain
                    manager = TemplateManager()
                    manager.load_from_yaml(yaml_file)
                    
                    # Register the domain
                    self.register_domain(domain, manager)
                    
                except Exception as e:
                    logger.error(f"Error loading templates from {yaml_file}: {str(e)}")
                    
            logger.info(f"Loaded templates from {len(yaml_files)} files in {directory}")
                
        except Exception as e:
            logger.error(f"Error scanning templates directory {directory}: {str(e)}")
    
    def save_templates_to_directory(self, directory: Optional[str] = None) -> None:
        """
        Save all templates to YAML files in a directory.
        
        This method saves each domain's templates to a separate YAML file
        in the specified directory.
        
        Args:
            directory: Directory path to save template files
                      (defaults to config/templates in the project root)
        """
        directory = directory or self.templates_dir
        
        try:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)
            
            for domain, manager in self.domain_managers.items():
                try:
                    # Save domain templates to a YAML file
                    yaml_file = os.path.join(directory, f"{domain}.yaml")
                    manager.save_to_yaml(yaml_file)
                    
                except Exception as e:
                    logger.error(f"Error saving templates for domain {domain}: {str(e)}")
                    
            logger.info(f"Saved templates for {len(self.domain_managers)} domains to {directory}")
                
        except Exception as e:
            logger.error(f"Error saving templates to directory {directory}: {str(e)}")
    
    def register_builtin_domains(self) -> None:
        """
        Register built-in template domains (translation, intention, etc.).
        """
        try:
            # Import built-in domain managers
            from templates.defaults import get_template_manager
            from src.templates.intention import get_intention_template_manager
            
            # Register domains
            self.register_domain('translation', get_template_manager())
            self.register_domain('intention', get_intention_template_manager())
            
            logger.info("Registered built-in template domains")
            
        except Exception as e:
            logger.error(f"Error registering built-in template domains: {str(e)}")


# Create global instance
registry = TemplateRegistry()
registry.register_builtin_domains()