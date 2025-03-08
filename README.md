# LLMPromptNexus

A unified framework for interacting with Large Language Models (LLMs) through a standardized interface. LLMPromptNexus simplifies working with multiple LLM providers while providing powerful templating and batching capabilities.

## 🚀 Quick Start

```bash
pip install llmprompt-nexus
```

```python
from llmprompt_nexus import NexusManager

# Initialize with your API keys
llm = NexusManager({
    "openai": "your-openai-key",
    "perplexity": "your-perplexity-key"
})

# Simple translation example
result = await llm.run_with_model(
    input_data={
        "text": "Hello world",
        "source_language": "English",
        "target_language": "Spanish"
    },
    model_id="sonar-pro",
    template_name="translation"
)
```

## 🌟 Key Features

- **Multiple LLM Providers**: Seamlessly work with OpenAI, Perplexity, and more through a single interface
- **Smart Template System**: Pre-built and custom templates for common NLP tasks
- **Efficient Batch Processing**: Handle large-scale operations with automatic rate limiting
- **Built-in Safety**: Automatic retries, rate limiting, and error handling

## 📦 Installation

### Using pip
```bash
pip install llmprompt-nexus
```

### From source
```bash
git clone https://github.com/EEstevanell/llmprompt-nexus.git
cd llmprompt-nexus
pip install -e .
```

## 🔑 Configuration

1. Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key"
export PERPLEXITY_API_KEY="your-key"
```

2. Or provide them during initialization:
```python
llm = UnifiedLLM({
    "openai": "your-openai-key",
    "perplexity": "your-perplexity-key"
})
```

## 📘 Basic Usage

### Single Operations

```python
# Simple text generation
response = await llm.generate(
    prompt="Explain quantum computing briefly",
    model_id="sonar-pro"
)

# Using templates
translation = await llm.run_with_model(
    input_data={
        "text": "Hello world",
        "source_language": "English",
        "target_language": "Spanish"
    },
    model_id="sonar-pro",
    template_name="translation"
)
```

### Batch Processing

```python
texts = ["First text", "Second text", "Third text"]
batch_inputs = [
    {
        "text": text,
        "source_language": "English",
        "target_language": "Spanish"
    }
    for text in texts
]

results = await llm.run_batch_with_model(
    input_data=batch_inputs,
    model_id="sonar-pro",
    template_name="translation"
)
```

## 🎯 Built-in Templates

- **Translation**: Convert text between languages
- **Classification**: Categorize text into predefined groups
- **Intent Detection**: Identify user intentions from text
- **Question Answering**: Generate answers based on context
- **Summarization**: Create concise text summaries

## ⚙️ Custom Templates

Create your own templates using YAML:

```yaml
templates:
  technical_qa:
    template: |
      Context: {context}
      Question: {question}
      Provide a technical answer based on the context.
    description: "Technical Q&A template"
    system_message: "You are a technical expert."
    required_variables: ["context", "question"]
```

## 📚 Documentation

For detailed documentation, visit our [documentation site](https://llmprompt-nexus.readthedocs.io/).

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means you are free to:
- Share and redistribute the material in any medium or format
- Adapt, remix, and transform the material

Under these conditions:
- **Attribution** — You must give appropriate credit when using this work, especially in academic research
- **NonCommercial** — You may not use the material for commercial purposes

For the full license text, see the [LICENSE](LICENSE) file.

