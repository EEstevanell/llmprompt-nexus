"""
LLMPrompt Nexus - A unified framework for interacting with Large Language Models (LLMs)

This package provides tools and utilities to simplify working with various
LLM providers through a common interface, with support for templating,
batching, and more advanced prompt engineering techniques.

For more information, see the documentation at:
https://llmprompt-nexus.readthedocs.io/
"""

__author__ = "Ernesto L. Estevanell-Valladares"
__maintainer__ = "LLMPromptNexus Contributors"
__email__ = "estevanell@etheria.eu"

# Import version information
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"

# Core imports that should be available at the top level
from .core.framework import LLMFramework
from .core.client_factory import ClientFactory

# Expose key modules for easier imports
from . import clients
from . import models
from . import templates
from . import processors
from . import utils

__all__ = [
    "LLMFramework",
    "ClientFactory",
    "clients",
    "models", 
    "templates",
    "processors",
    "utils",
]