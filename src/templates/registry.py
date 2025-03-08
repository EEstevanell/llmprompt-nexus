"""
Central template registry system for the UnifiedLLM framework.

The template registry provides a centralized location to register,
discover and manage templates across the framework.
"""
from typing import Dict, Optional
from pathlib import Path

from src.templates.base import Template
from src.templates.manager import TemplateManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TemplateRegistry:
    """
    Global registry for templates across the framework.
    Templates are loaded from YAML files in config/templates/.
    """
    
    def __init__(self):
        self.manager = TemplateManager()
        
    def load_all_templates(self, config_dir: Optional[Path] = None):
        """Load all templates from the config directory."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / 'config' / 'templates'
            
        if not config_dir.exists():
            logger.warning(f"Template config directory {config_dir} does not exist")
            return
            
        # Load from YAML files
        self.manager = TemplateManager.from_yaml_dir(config_dir)
        logger.info(f"Loaded {len(self.manager.templates)} templates from {config_dir}")
    
    def get_template(self, name: str) -> Template:
        """Get a template by name."""
        return self.manager.get_template(name)
    
    def register_template(self, template: Template):
        """Register a new template."""
        self.manager.register_template(template)
    
    def register_templates(self, templates: Dict[str, Template]):
        """Register multiple templates."""
        self.manager.register_templates(templates)

# Create global instance and load templates
registry = TemplateRegistry()
registry.load_all_templates()