"""
Template selection strategies for different domains.
"""
from typing import Dict, Any
from src.templates.base import Template, TemplateStrategy

class TranslationTemplateStrategy(TemplateStrategy):
    """Strategy for selecting and preparing translation templates."""
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """Select appropriate translation template based on content type."""
        content_type = input_data.get('content_type', 'default')
        if content_type == 'technical':
            return templates.get('technical', templates['default'])
        elif content_type == 'literary':
            return templates.get('literary', templates['default'])
        return templates['default']
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """Prepare variables for translation template."""
        required_vars = template.get_required_variables()
        
        # Map common variable names
        variables = {
            'text': input_data.get('text', input_data.get('content', '')),
            'source_language': input_data.get('source_lang', input_data.get('from_lang', 'English')),
            'target_language': input_data.get('target_lang', input_data.get('to_lang', ''))
        }
        
        # Add any additional variables
        for var in required_vars:
            if var not in variables and var in input_data:
                variables[var] = input_data[var]
                
        return variables

class SummarizationTemplateStrategy(TemplateStrategy):
    """Strategy for selecting and preparing summarization templates."""
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """Select appropriate summarization template based on content type."""
        # Choose template based on context
        if input_data.get('academic') or input_data.get('scholarly'):
            return templates.get('academic', templates['default'])
        elif input_data.get('business') or input_data.get('executive'):
            return templates.get('executive', templates['default'])
        return templates['default']
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """Prepare variables for summarization template."""
        variables = {
            'text': input_data.get('text', input_data.get('content', '')),
            'length': input_data.get('length', 'concise')
        }
        
        # Add any additional required variables
        for var in template.get_required_variables():
            if var not in variables and var in input_data:
                variables[var] = input_data[var]
                
        return variables

class AnalysisTemplateStrategy(TemplateStrategy):
    """Strategy for general text analysis templates."""
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """Select appropriate analysis template based on analysis focus."""
        analysis_focus = input_data.get('focus', 'default')
        
        if analysis_focus == 'style':
            return templates.get('style', templates['default'])
        elif analysis_focus == 'structure':
            return templates.get('structure', templates['default'])
        return templates['default']
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """Prepare variables for analysis template."""
        variables = {
            'text': input_data.get('text', input_data.get('content', '')),
        }
        
        # Add any additional required variables
        for var in template.get_required_variables():
            if var not in variables and var in input_data:
                variables[var] = input_data[var]
                
        return variables

class SentimentTemplateStrategy(TemplateStrategy):
    """Strategy for sentiment analysis templates."""
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """Select appropriate sentiment analysis template."""
        if input_data.get('aspects'):
            return templates.get('detailed', templates['default'])
        elif input_data.get('comparative', False):
            return templates.get('comparative', templates['default'])
        return templates['default']
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """Prepare variables for sentiment analysis template."""
        variables = {
            'text': input_data.get('text', input_data.get('content', '')),
        }
        
        # Add sentiment-specific variables
        if input_data.get('aspects'):
            variables['aspects'] = input_data['aspects']
        
        # Add any additional required variables
        for var in template.get_required_variables():
            if var not in variables and var in input_data:
                variables[var] = input_data[var]
                
        return variables

class TechnicalTemplateStrategy(TemplateStrategy):
    """Strategy for technical analysis templates."""
    
    def select_template(self, input_data: Dict[str, Any], templates: Dict[str, Template]) -> Template:
        """Select appropriate technical analysis template."""
        content_type = input_data.get('type', 'default')
        
        if content_type == 'architecture':
            return templates.get('architecture', templates['default'])
        elif content_type == 'code':
            return templates.get('code', templates['default'])
        return templates['default']
    
    def prepare_variables(self, input_data: Dict[str, Any], template: Template) -> Dict[str, Any]:
        """Prepare variables for technical analysis template."""
        variables = {
            'text': input_data.get('text', input_data.get('content', '')),
        }
        
        # Add technical-specific variables
        if input_data.get('focus_areas'):
            variables['focus_areas'] = input_data['focus_areas']
        if input_data.get('tech_stack'):
            variables['tech_stack'] = input_data['tech_stack']
        
        # Add any additional required variables
        for var in template.get_required_variables():
            if var not in variables and var in input_data:
                variables[var] = input_data[var]
                
        return variables