"""
Base template system for the UnifiedLLM framework.
"""
from typing import Dict, Any, Optional, List, Set
import re
from abc import ABC, abstractmethod

class Template:
    """
    Base template class that represents a template with placeholders.
    
    This class manages the template text, required variables, and handles
    rendering the template with provided values.
    """
    
    def __init__(self, template_text: str, name: str = "unnamed", description: str = ""):
        """
        Initialize a template with template text and metadata.
        
        Args:
            template_text: The template text with {placeholder} variables
            name: Name of the template
            description: Description of what the template does
        """
        self.template_text = template_text
        self.name = name
        self.description = description
        self._required_vars = self._extract_variables(template_text)
        
    def _extract_variables(self, template_text: str) -> Set[str]:
        """
        Extract variable names from the template text.
        
        Args:
            template_text: The template text with {placeholder} variables
            
        Returns:
            Set of variable names without the braces
        """
        # Find all patterns like {variable_name} in the template text
        pattern = r"\{([a-zA-Z0-9_]+)\}"
        matches = re.finditer(pattern, template_text)
        
        # Extract the variable names without the braces
        variables = {match.group(1) for match in matches}
        return variables
    
    def get_required_variables(self) -> Set[str]:
        """
        Get the required variables for this template.
        
        Returns:
            Set of variable names that need to be provided to render the template
        """
        return self._required_vars
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """
        Validate that all required variables are provided.
        
        Args:
            variables: Dictionary of variable names to values
            
        Returns:
            List of missing variable names, empty if all required variables are provided
        """
        provided_vars = set(variables.keys())
        missing_vars = self._required_vars - provided_vars
        return list(missing_vars)
    
    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render the template with provided variables.
        
        Args:
            variables: Dictionary of variable names to values
            
        Returns:
            The rendered template text
            
        Raises:
            ValueError: If required variables are missing
        """
        missing_vars = self.validate_variables(variables)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
            
        return self.template_text.format(**variables)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary representation.
        
        Returns:
            Dictionary with template metadata and text
        """
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template_text,
            "required_variables": list(self._required_vars)
        }


class TemplateStrategy(ABC):
    """
    Abstract base class for template strategies.
    
    Template strategies determine how to select and apply templates based
    on input data and the current context.
    """
    
    @abstractmethod
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """
        Select the appropriate template based on input data.
        
        Args:
            input_data: The input data containing intention or purpose
            templates: Dictionary of available templates
            
        Returns:
            Selected Template instance
            
        Raises:
            ValueError: If no suitable template is found
        """
        pass
    
    @abstractmethod
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """
        Prepare variables for template rendering based on input data.
        
        Args:
            input_data: The input data with values for template variables
            template: The template to render
            
        Returns:
            Dictionary of variables to use for template rendering
            
        Raises:
            ValueError: If required variables cannot be prepared
        """
        pass


class IntentionTemplateStrategy(TemplateStrategy):
    """
    Strategy for selecting templates based on an 'intention' field.
    
    This strategy selects a template by matching the 'intention' field
    in the input data with a template key.
    """
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """
        Select template based on 'intention' field in input data.
        
        Args:
            input_data: Input data with 'intention' field
            templates: Available templates
            
        Returns:
            Selected Template instance
            
        Raises:
            ValueError: If no matching template is found or 'intention' field is missing
        """
        if 'intention' not in input_data:
            if 'default' in templates:
                return templates['default']
            raise ValueError("Input data missing 'intention' field and no default template available")
            
        intention = input_data['intention']
        
        if intention not in templates:
            if 'default' in templates:
                return templates['default']
            raise ValueError(f"No template found for intention '{intention}' and no default template available")
            
        return templates[intention]
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """
        Prepare variables for template rendering by directly using input data.
        
        Args:
            input_data: Input data with variable values
            template: The template to render
            
        Returns:
            Variables to use for template rendering
            
        Raises:
            ValueError: If required variables are missing
        """
        required_vars = template.get_required_variables()
        
        # Check for missing variables
        missing_vars = [var for var in required_vars if var not in input_data]
        if missing_vars:
            raise ValueError(f"Missing required variables for template: {', '.join(missing_vars)}")
            
        # Only include the variables required by the template
        variables = {key: input_data[key] for key in required_vars}
        
        return variables


class TranslationTemplateStrategy(TemplateStrategy):
    """
    Specialized strategy for translation templates.
    """
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """
        Select template for translation tasks.
        
        Args:
            input_data: Input data for translation task
            templates: Available templates
            
        Returns:
            Translation template
            
        Raises:
            ValueError: If translation template is not found
        """
        if 'translate' in templates:
            return templates['translate']
        if 'default' in templates:
            return templates['default']
        raise ValueError("No translation or default template found")
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """
        Prepare variables specific to translation templates.
        
        Args:
            input_data: Translation task data with text, source_lang, target_lang
            template: The template to render
            
        Returns:
            Variables to use for template rendering
            
        Raises:
            ValueError: If required variables are missing
        """
        required_vars = template.get_required_variables()
        
        # Create a variables map with all input data
        variables = {key: value for key, value in input_data.items() if key in required_vars}
        
        # Check for missing variables
        missing_vars = [var for var in required_vars if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables for translation template: {', '.join(missing_vars)}")
        
        return variables