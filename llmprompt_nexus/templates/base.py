"""
Base template system for the UnifiedLLM framework.
"""
from typing import Dict, Any, Optional, List, Set
import re

class Template:
    """Base template class that represents a template with placeholders."""
    
    def __init__(self, 
                 template_text: str, 
                 name: str = "unnamed", 
                 description: str = "",
                 system_message: Optional[str] = None,
                 required_variables: Optional[Set[str]] = None):
        """
        Initialize a template.
        
        Args:
            template_text: The template text with {variable} placeholders
            name: Name of the template
            description: Description of what the template does
            system_message: Optional system message for chat models
            required_variables: Optional set of required variables. If not provided,
                             will be extracted from template_text
        """
        self.validate_template_text(template_text)
        self.template_text = template_text
        self.name = name
        self.description = description
        self.system_message = system_message
        self._required_vars = required_variables or self._extract_variables(template_text)
    
    @staticmethod
    def validate_template_text(template_text: str) -> None:
        """Validate template text format and content."""
        if not isinstance(template_text, str):
            raise ValueError("Template text must be a string")
        if not template_text.strip():
            raise ValueError("Template text cannot be empty")
        
        # Check for basic formatting issues
        if "{" not in template_text or "}" not in template_text:
            raise ValueError("Template must contain at least one variable placeholder {var}")

    @staticmethod
    def validate_template_config(config: Dict[str, Any]) -> None:
        """
        Validate a template configuration dictionary.
        
        Args:
            config: Dictionary containing template configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = {"template"}
        optional_fields = {"name", "description", "system_message", "required_variables"}
        
        if not isinstance(config, dict):
            raise ValueError("Template configuration must be a dictionary")
            
        # Check for required fields
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in template config: {missing_fields}")
            
        # Check for unknown fields
        unknown_fields = set(config.keys()) - (required_fields | optional_fields)
        if unknown_fields:
            raise ValueError(f"Unknown fields in template config: {unknown_fields}")
            
        # Validate field types
        if not isinstance(config["template"], str):
            raise ValueError("Template text must be a string")
        if "name" in config and not isinstance(config["name"], str):
            raise ValueError("Template name must be a string")
        if "description" in config and not isinstance(config["description"], str):
            raise ValueError("Template description must be a string")
        if "system_message" in config and not isinstance(config["system_message"], str):
            raise ValueError("System message must be a string")
        if "required_variables" in config:
            if not isinstance(config["required_variables"], (list, set)):
                raise ValueError("Required variables must be a list or set")
            if not all(isinstance(v, str) for v in config["required_variables"]):
                raise ValueError("Required variables must be strings")

    def _extract_variables(self, template_text: str) -> Set[str]:
        """Extract variable names from the template text."""
        pattern = r"\{([a-zA-Z0-9_]+)\}"
        matches = re.finditer(pattern, template_text)
        return {match.group(1) for match in matches}
    
    def get_required_variables(self) -> Set[str]:
        """Get the required variables for this template."""
        return self._required_vars
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Get list of missing required variables."""
        if not isinstance(variables, dict):
            raise ValueError("Variables must be provided as a dictionary")
            
        missing = []
        for var in self._required_vars:
            if var not in variables:
                missing.append(var)
            elif variables[var] is None:
                missing.append(var)
        return missing
    
    def prepare_variables(self, input_data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare and validate variables from input data.
        
        Args:
            input_data: Input data containing variables
            defaults: Optional default values for variables
            
        Returns:
            Dictionary of prepared variables
            
        Raises:
            ValueError: If required variables are missing
        """
        variables = {}
        
        # Start with defaults if provided
        if defaults:
            variables.update(defaults)
            
        # Add input variables, overriding defaults
        variables.update({
            k: v for k, v in input_data.items() 
            if k in self._required_vars
        })
        
        # Validate
        missing = self.validate_variables(variables)
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")
            
        return variables
    
    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render the template with provided variables.
        
        Args:
            variables: Dictionary of variables to use in template
            
        Returns:
            Rendered template text
            
        Raises:
            ValueError: If required variables are missing
        """
        try:
            return self.template_text.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
        except Exception as e:
            raise ValueError(f"Error rendering template: {e}")
    
    def get_messages(self, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Get formatted messages list including system message if present.
        
        Args:
            variables: Dictionary of variables to use in template
            
        Returns:
            List of message dictionaries for chat models
        """
        messages = []
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        messages.append({
            "role": "user",
            "content": self.render(variables)
        })
        return messages