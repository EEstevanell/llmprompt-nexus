# UnifiedLLM Framework

A flexible framework for interacting with multiple Language Model APIs through a unified interface. This framework provides template-based interactions, rate limiting, and batch processing capabilities across different LLM providers.

## Project Structure

The project structure of `chat-completions` is as follows:

```
chat-completions/
├── main.py
├── clients/
│   ├── __init__.py
│   ├── base.py
│   ├── perplexity.py
│   └── openai.py
├── models/
│   ├── __init__.py
│   ├── config.py
│   └── registry.py
├── rate_limiting/
│   ├── __init__.py
│   └── limiter.py
├── templates/
│   ├── __init__.py
│   └── intention.py
└── utils/
    ├── __init__.py
    └── async_utils.py
```

## Features

- **Multi-Provider Support**: Unified interface for OpenAI, Perplexity and other LLM providers
- **Template System**: Flexible template management for different use cases
- **API Management**: Built-in rate limiting and key validation
- **Batch Processing**: Optimized batch operations where supported
- **Async Processing**: Asynchronous request handling
- **Extensible**: Easy to add new providers and templates

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/EEstevanell/chat-completions.git
   cd chat-completions
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:
   ```bash
   export PERPLEXITY_API_KEY="your_perplexity_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage

To use the project, follow these steps:

1. Place your input TSV files in the `inputs/` directory.

2. Modify the `main.py` file to select the models you want to use:
   ```python
   models_to_run = ["gpt-4o-mini", "sonar-small", "sonar-huge"]
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

   The script will process each input file with the selected models and save the results in the same directory as the input files.

## Components

### Clients

The `clients/` directory contains the API clients for different services:

- `base.py`: Abstract base class for API clients
- `perplexity.py`: Client for the Perplexity API
- `openai.py`: Client for the OpenAI API

### Models

The `models/` directory handles model configuration and registration:

- `config.py`: Defines the `ModelConfig` class
- `registry.py`: Manages the registration and retrieval of model configurations

### Rate Limiting

The `rate_limiting/` directory contains the rate limiting implementation:

- `limiter.py`: Implements the `RateLimiter` class to manage API request rates

### Templates

The `templates/` directory stores the templates for different use cases:

- `intention.py`: Contains templates for intention detection

### Utils

The `utils/` directory includes utility functions:

- `async_utils.py`: Contains the `ProcessingManager` class for asynchronous processing

## Customization

To add new models or APIs:

1. Create a new client in the `clients/` directory.

2. Register the new model in `models/registry.py`.

3. Update the `main.py` file to include the new model in `models_to_run`.

To modify templates:

- Edit or add new templates in `templates/intention.py`.

## License

This project is licensed under the MIT License.

