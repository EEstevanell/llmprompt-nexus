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
    from different domains, supporting both default and user-defined templates.
    """
    
    def __init__(self):
        """Initialize the template registry with empty managers."""
        # Dictionary mapping domain names to TemplateManager instances
        self.domain_managers: Dict[str, TemplateManager] = {}
        
        # Default templates directory (inside package)
        self.default_templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config", "templates")
        
        # User templates directory (in user's config)
        self.user_templates_dir = os.path.join(os.path.expanduser("~"), ".config", "unifiedllm", "templates")
        
    def register_domain(self, domain: str, manager: TemplateManager, is_default: bool = True) -> None:
        """
        Register a template manager for a specific domain.
        
        Args:
            domain: The domain name (e.g., 'translation', 'intention')
            manager: The template manager for this domain
            is_default: Whether these are default templates
        """
        if domain in self.domain_managers and is_default:
            # Don't override user templates with defaults
            return
            
        logger.info(f"Registering {'default' if is_default else 'user'} template domain: {domain}")
        self.domain_managers[domain] = manager
        
    def get_domain(self, domain: str) -> TemplateManager:
        """Get template manager for a domain."""
        if domain not in self.domain_managers:
            raise ValueError(f"Domain not found: {domain}")
        return self.domain_managers[domain]
        
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self.domain_managers.keys())
        
    def load_all_templates(self) -> None:
        """Load all templates from both default and user directories."""
        # First load default templates
        self.load_templates_from_directory(self.default_templates_dir, is_default=True)
        
        # Then load user templates (will override defaults if same domain)
        if os.path.exists(self.user_templates_dir):
            self.load_templates_from_directory(self.user_templates_dir, is_default=False)
        
    def load_templates_from_directory(self, directory: str, is_default: bool = True) -> None:
        """
        Load templates from YAML files in a directory.
        
        Args:
            directory: Directory path to scan for template files
            is_default: Whether these are default templates
        """
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
                    self.register_domain(domain, manager, is_default=is_default)
                    
                except Exception as e:
                    logger.error(f"Error loading templates from {yaml_file}: {str(e)}")
                    
            logger.info(f"Loaded templates from {len(yaml_files)} files in {directory}")
                
        except Exception as e:
            logger.error(f"Error scanning templates directory {directory}: {str(e)}")
            
    def get_template(self, domain: str, template_name: str) -> Template:
        """Get a specific template from a domain."""
        return self.get_domain(domain).get_template(template_name)
        
    def list_templates(self, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available templates, optionally filtered by domain.
        
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
        
    def create_user_template_dir(self) -> None:
        """Create user template directory if it doesn't exist."""
        os.makedirs(self.user_templates_dir, exist_ok=True)
        
    def save_user_template(self, domain: str, template: Template) -> None:
        """
        Save a user-defined template.
        
        Args:
            domain: Template domain
            template: Template to save
        """
        self.create_user_template_dir()
        
        # Get or create domain manager
        if domain not in self.domain_managers:
            self.domain_managers[domain] = TemplateManager()
        manager = self.domain_managers[domain]
        
        # Register template
        manager.register_template(template)
        
        # Save to file
        yaml_file = os.path.join(self.user_templates_dir, f"{domain}.yaml")
        manager.save_to_yaml(yaml_file)

# Create global instance and load templates
registry = TemplateRegistry()
registry.load_all_templates()