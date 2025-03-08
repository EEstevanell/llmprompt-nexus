"""
Central template registry system for the UnifiedLLM framework.
"""
from typing import Dict, Optional, List
from pathlib import Path

from llmprompt_nexus.templates.base import Template
from llmprompt_nexus.templates.manager import TemplateManager
from llmprompt_nexus.utils.logger import get_logger

logger = get_logger(__name__)

class TemplateRegistry:
    """
    Global registry for templates across the framework.
    Templates are loaded from YAML files in config/templates/.
    """
    
    # Core template types supported by the framework
    TEMPLATE_TYPES = [
        'classification',
        'intent',
        'qa',
        'summarization',
        'translation'
    ]
    
    def __init__(self):
        self.manager = TemplateManager()
        self._type_managers: Dict[str, TemplateManager] = {}
        
    def load_all_templates(self, config_dir: Optional[Path] = None):
        """Load all templates from the config directory."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / 'config' / 'templates'
            
        if not config_dir.exists():
            logger.warning(f"Template config directory {config_dir} does not exist")
            return
            
        # Load all templates into main manager
        self.manager = TemplateManager.from_yaml_dir(config_dir)
        logger.info(f"Loaded {len(self.manager.templates)} templates from {config_dir}")
        
        # Load type-specific managers
        for template_type in self.TEMPLATE_TYPES:
            type_file = config_dir / f"{template_type}.yaml"
            if type_file.exists():
                try:
                    self._type_managers[template_type] = TemplateManager.from_yaml(type_file)
                except Exception as e:
                    logger.error(f"Error loading {template_type} templates: {str(e)}")
    
    def get_template(self, name: str) -> Template:
        """Get a template by name."""
        return self.manager.get_template(name)
    
    def get_template_manager(self, template_type: str = 'translation') -> TemplateManager:
        """
        Get template manager for specified template type.
        
        Args:
            template_type: Type of templates to load (e.g., 'translation', 'qa')
            
        Returns:
            TemplateManager with templates from the specified type
        """
        if template_type not in self.TEMPLATE_TYPES:
            raise ValueError(f"Unknown template type: {template_type}")
            
        if template_type not in self._type_managers:
            raise ValueError(f"No templates loaded for type: {template_type}")
            
        return self._type_managers[template_type]
    
    def list_template_types(self) -> List[str]:
        """List available template types."""
        return list(self._type_managers.keys())
    
    def register_template(self, template: Template):
        """Register a new template."""
        self.manager.register_template(template)
    
    def register_templates(self, templates: Dict[str, Template]):
        """Register multiple templates."""
        self.manager.register_templates(templates)

# Create global instance and load templates
registry = TemplateRegistry()
registry.load_all_templates()