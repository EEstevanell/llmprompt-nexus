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
        self.template_text = template_text
        self.name = name
        self.description = description
        self.system_message = system_message
        self._required_vars = required_variables or self._extract_variables(template_text)
    
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
        return [var for var in self._required_vars if var not in variables]
    
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
        missing = self.validate_variables(variables)
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")
        return self.template_text.format(**variables)
    
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