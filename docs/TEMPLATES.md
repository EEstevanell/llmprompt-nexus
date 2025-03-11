# LLMPrompt-Nexus Templates Guide

This guide explains how to use the template system in LLMPrompt-Nexus for interacting with language models. Templates provide a structured way to format prompts and control model behavior.

## Table of Contents
- [Understanding Templates](#understanding-templates)
- [Built-in Templates](#built-in-templates)
  - [Default Template](#default-template)
  - [Translation Template](#translation-template)
  - [Summarization Template](#summarization-template)
  - [Classification Template](#classification-template)
  - [Question Answering Template](#question-answering-template)
  - [Intent Detection Template](#intent-detection-template)
- [Custom Templates](#custom-templates)
  - [Dictionary-based Templates](#1-dictionary-based-templates)
  - [YAML-based Templates](#2-yaml-based-templates)
- [Templates in Batch Processing](#templates-in-batch-processing)
- [Advanced Template Features](#advanced-template-features)
- [Best Practices](#best-practices)

## Understanding Templates

Templates in LLMPrompt-Nexus are pre-defined structures that help format your requests to language models. Think of them as fill-in-the-blank forms where you provide specific information (variables), and the template arranges that information in a way that gets the best responses from the AI model.

A template typically consists of:
- **The template text**: The actual prompt with placeholders for variables
- **System message**: Instructions that set the tone and behavior of the AI model
- **Required variables**: The information you need to provide

## Built-in Templates

LLMPrompt-Nexus comes with several ready-to-use templates for common tasks. Here's what each one does, what variables they need, and how to use them:

### Default Template

This is the simplest template used when you just want to send a plain text prompt.

**Template Content:**
```
{prompt}
```

**System Message:**
```
You are a helpful AI assistant. Provide accurate, informative, and concise responses.
```

**Required Variables:**
- `prompt`: Your message or question to the AI

**Example Usage:**
```python
from llmprompt_nexus import NexusManager

# Initialize the framework
api_keys = {
    "openai": "your-api-key-here"
}
framework = NexusManager(api_keys)

# Simple prompt
result = await framework.generate(
    input_data="What is machine learning?",
    model_id="sonar"  # Replace with your preferred model
)

print(f"Response: {result.get('response')}")
```

### Translation Template

Translates text from one language to another.

**Template Content:**
```
Translate the following text from {source_language} to {target_language}:

{text}

Translation:
```

**System Message:**
```
You are an expert translator with deep knowledge of both source and target languages. 
You excel at preserving meaning, style, and technical accuracy in translations.
```

**Required Variables:**
- `text`: The text you want to translate
- `source_language`: The language of the original text (e.g., "English")
- `target_language`: The language you want to translate to (e.g., "Spanish")

**Example Usage:**
```python
input_data = {
    "text": "Hello world",
    "source_language": "English",
    "target_language": "Spanish"
}

result = await framework.generate(
    input_data=input_data,
    model_id="sonar",
    template_name="translation"
)

print(f"Translation: {result.get('response')}")
# Expected output might be: "Hola mundo"
```

### Summarization Template

Generates a summary of provided text.

**Template Content:**
```
Summarize the following text:

{text}

Length: {length}

Provide:
- Concise summary
- Key points
- Important details preserved
```

**System Message:**
```
You are an expert in text summarization, capable of extracting and condensing 
the most important information while maintaining accuracy and coherence.
```

**Required Variables:**
- `text`: The longer text you want summarized
- `length`: How long you want the summary to be (e.g., "3 sentences", "100 words", "brief")

**Example Usage:**
```python
input_data = {
    "text": """
    The Python programming language was created by Guido van Rossum and was first released in 1991.
    It emphasizes code readability with its notable use of significant whitespace. Python features a
    dynamic type system and automatic memory management and supports multiple programming paradigms.
    """,
    "length": "2 sentences"
}

result = await framework.generate(
    input_data=input_data,
    model_id="sonar",
    template_name="summarization"
)

print(f"Summary: {result.get('response')}")
```

### Classification Template

Categorizes text into predefined categories.

**Template Content:**
```
Classify the following text into one of the given categories:

Text: {text}
Categories: {categories}

Provide:
- Selected category
- Confidence score (0-1)
- Reasoning for classification
```

**System Message:**
```
You are an expert in text classification, capable of accurately categorizing 
content based on semantic meaning and context.
```

**Required Variables:**
- `text`: The text to be classified
- `categories`: A list of possible categories (e.g., ["positive", "negative", "neutral"])

**Example Usage:**
```python
input_data = {
    "text": "I absolutely loved this product! Best purchase ever.",
    "categories": ["positive", "negative", "neutral"]
}

result = await framework.generate(
    input_data=input_data, 
    model_id="sonar",
    template_name="classification"
)

print(f"Classification: {result.get('response')}")
# Expected output might include the category "positive" with a high confidence score
```

### Question Answering Template

Answers questions based on provided context.

**Template Content:**
```
Answer the following question based on the given context:

Context: {context}
Question: {question}

Provide:
- Direct answer
- Confidence score (0-1)
- Supporting evidence from context
```

**System Message:**
```
You are an expert in reading comprehension and question answering, 
capable of extracting precise answers from given contexts.
```

**Required Variables:**
- `context`: The background information or text that contains the answer
- `question`: The question to be answered based on the context

**Example Usage:**
```python
input_data = {
    "context": """
    The UnifiedLLM framework provides a consistent interface for working with different
    Language Model APIs. It supports template-based interactions, rate limiting,
    batch processing, and multiple providers.
    """,
    "question": "What are the main features of the UnifiedLLM framework?"
}

result = await framework.generate(
    input_data=input_data,
    model_id="sonar",
    template_name="qa"
)

print(f"Answer: {result.get('response')}")
```

### Intent Detection Template

Identifies the intent of a given text from a list of possible intents.

**Template Content:**
```
Detect the intent in the following text:

{text}

Provide:
- Primary intent
- Secondary intents (if any)
- Confidence score (0-1)
- Key intent indicators
```

**System Message:**
```
You are an expert in understanding user intentions and goals from natural language, 
capable of identifying both explicit and implicit intents.
```

**Required Variables:**
- `text`: The user message or text to analyze for intent

**Example Usage:**
```python
input_data = {
    "text": "Can you help me book a flight to New York?",
    "possible_intents": ["booking", "information", "support", "other"]
}

result = await framework.generate(
    input_data=input_data,
    model_id="sonar",
    template_name="intent"
)

print(f"Intent: {result.get('response')}")
# Expected output might identify "booking" as the primary intent
```

## Custom Templates

When the built-in templates don't fit your specific needs, you can create custom templates in two ways:

### 1. Dictionary-based Templates

Create templates directly in your Python code using dictionaries:

**Template Structure:**
```python
template_config = {
    "template": "Your template text with {variable1} and {variable2}",
    "name": "template_name",  # Optional
    "description": "What the template does",  # Optional
    "system_message": "Instructions for the AI model",  # Optional
    "required_variables": ["variable1", "variable2"]  # Optional
}
```

**Example: Custom Technical Q&A Template**
```python
# Define a custom template
qa_template = {
    "template": """Context: {context}

Question: {question}

Please provide a clear, technical answer based on the context above.""",
    "name": "technical_qa",
    "description": "Template for technical question answering",
    "system_message": "You are a technical expert. Provide accurate, technical answers based on the given context.",
    "required_variables": ["context", "question"]
}

# Use the template
qa_input = {
    "context": """
    The rate limiter implements a token bucket algorithm with a configurable bucket size and refill rate.
    Tokens are consumed for each API request and automatically refilled over time.
    When the bucket is empty, requests are delayed until enough tokens are available.
    """,
    "question": "How does the rate limiter handle requests when the token bucket is empty?"
}

result = await framework.generate(
    input_data=qa_input,
    model_id="sonar",
    template_config=qa_template  # Pass the template configuration directly
)

print(f"Answer: {result.get('response')}")
```

### 2. YAML-based Templates

For more permanent templates that you'll use often, you can create YAML files:

**Template Structure (YAML file):**
```yaml
template: |
  Your template text with {variable1} and {variable2}
name: template_name
description: What the template does
system_message: Instructions for the AI model
required_variables:
  - variable1
  - variable2
```

**Example: Academic Summary Template (`academic_summary.yaml`):**
```yaml
template: |
  Please provide a {style} summary of the following text, limited to {max_length} words:

  {text}
name: academic_summary
description: Template for academic-style summarization
system_message: You are an academic writing assistant specializing in clear, concise summaries.
required_variables:
  - text
  - style
  - max_length
```

**Using the YAML template:**
```python
# Initialize with custom templates directory
framework = NexusManager(
    api_keys=api_keys,
    templates_dir="/path/to/custom/templates"  # Directory containing your YAML template files
)

summary_input = {
    "text": "Your text here...",
    "style": "academic",
    "max_length": 50
}

result = await framework.generate(
    input_data=summary_input,
    model_id="sonar",
    template_name="academic_summary"  # Will load from your custom template directory
)
```

## Templates in Batch Processing

When you need to process multiple inputs using the same template, batch processing is more efficient:

```python
# Define a custom template for batch processing
batch_translation_template = {
    "template": "Translate the following text from {source_language} to {target_language}:\n\n{text}",
    "description": "Optimized translation template for batch processing",
    "system_message": "You are an expert translator specializing in batch translation tasks."
}

# Create batch input data - each dictionary is a separate translation task
batch_inputs = [
    {
        "text": "The API implements robust rate limiting mechanisms.",
        "source_language": "English",
        "target_language": "Spanish"
    },
    {
        "text": "Data structures are optimized for concurrent access.",
        "source_language": "English", 
        "target_language": "Spanish"
    },
    # Add more items as needed
]

# Process all translations at once
results = await framework.generate_batch(
    inputs=batch_inputs,
    model_id="sonar",
    template_config=batch_translation_template
)

# Display results
for i, result in enumerate(results):
    print(f"Translation {i+1}: {result.get('response')}")
```

## Advanced Template Features

### System Messages

System messages help guide the AI model's behavior and expertise. Think of them as instructions on how the AI should approach your request:

```python
template_config = {
    "template": "Explain the concept of {topic} in simple terms.",
    "system_message": "You are an expert educator who specializes in explaining complex topics to beginners. Your explanations should be clear, concise, and use simple analogies."
}
```

### Required Variables

When you specify required variables, the library will check if all necessary information is provided before sending the request:

```python
template_config = {
    "template": "Generate a {length} {style} story about {topic}.",
    "required_variables": ["length", "style", "topic"]
}
```

If you try to use this template without providing all required variables, the system will return an error, helping you catch mistakes early.

### Template Chaining

For complex tasks, you can process text through multiple templates in sequence:

```python
# First template generates a summary
summary_result = await framework.generate(
    input_data={"text": long_text},
    model_id="sonar",
    template_name="summarization"
)

# Second template analyzes the summary
analysis_result = await framework.generate(
    input_data={"text": summary_result.get('response')},
    model_id="sonar",
    template_name="analysis"
)
```

### File Processing with Templates

Process data from files using templates:

```python
output_path = await framework.process_file(
    file_path="./input_data.tsv",  # Tab-separated file with headers
    model_id="sonar",
    template_config={
        "template": "Analyze the sentiment of the following text:\n\n{text}",
        "system_message": "You are an expert at sentiment analysis."
    }
)

print(f"Results saved to: {output_path}")
```

The input file should be structured with headers that match your template variables (e.g., a column named "text" for the {text} variable).

## Best Practices

1. **Keep templates focused on specific tasks**: Each template should have one clear purpose.
   
2. **Use descriptive variable names**: Choose names like `customer_feedback` rather than just `text` to make your templates more self-documenting.
   
3. **Provide detailed system messages**: This helps the AI model understand its role and how to respond.
   
4. **Always define required variables**: This prevents errors from missing information.
   
5. **Test your templates with various inputs**: Make sure they work as expected with different kinds of content.
   
6. **Include examples in complex templates**: For templates that need specific formatting, include examples of what you expect.
   
7. **Use comments in your templates**: Add explanations about what different parts do.
   
8. **Consider the model's strengths**: Different models may perform better with slightly different template structures.

9. **Start simple and iterate**: Begin with a basic template and refine it based on the results.

10. **Keep a library of successful templates**: Save templates that work well for reuse in future projects.

For more examples, see the `examples/` directory in the repository.