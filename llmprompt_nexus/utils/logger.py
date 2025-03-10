# src/utils/logging.py
import logging
from enum import IntEnum
from typing import Optional, Union
import sys

class VerboseLevel(IntEnum):
    NONE = 0
    ERRORS = 1
    WARNINGS = 2
    INFO = 3
    DEBUG = 4
    
level_mapping = {
        VerboseLevel.NONE: logging.CRITICAL + 1,
        VerboseLevel.ERRORS: logging.ERROR,
        VerboseLevel.WARNINGS: logging.WARNING,
        VerboseLevel.INFO: logging.INFO,
        VerboseLevel.DEBUG: logging.DEBUG
    }

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        logging.DEBUG: '\033[0;36m',    # Cyan
        logging.INFO: '\033[0;32m',     # Green
        logging.WARNING: '\033[0;33m',  # Yellow
        logging.ERROR: '\033[0;31m',    # Red
        logging.CRITICAL: '\033[0;35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelno)
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        record.msg = f'{color}{record.msg}{self.RESET}'
        return super().format(record)

_framework_logger = None

def configure_logger(level: Union[str, int] = "INFO") -> None:
    """
    Configure the framework-wide logger with the specified log level.
    This should be called once when initializing NexusManager.
    
    Args:
        level: Logging level as string or integer (e.g., "INFO" or logging.INFO)
    """
    global _framework_logger
    
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger if it doesn't exist
    if _framework_logger is None:
        _framework_logger = logging.getLogger("llmprompt_nexus")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        _framework_logger.addHandler(handler)
    
    # Set level for the framework logger and all its children
    _framework_logger.setLevel(level)

def get_logger(
    name: str,
    level: Optional[VerboseLevel] = VerboseLevel.INFO
) -> logging.Logger:
    """
    Creates a logger with the specified verbosity level.
    
    Args:
        name: Logger name (typically __name__)
        level: Verbosity level from VerboseLevel enum
    """
    if (level < VerboseLevel.NONE) or (level > VerboseLevel.DEBUG):
        raise ValueError("Invalid verbose level. Must be between None and Debug.")
    
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(console_handler)
    
    # Map VerboseLevel to logging levels
    level_mapping = {
        VerboseLevel.NONE: logging.CRITICAL + 1,
        VerboseLevel.ERRORS: logging.ERROR,
        VerboseLevel.WARNINGS: logging.WARNING,
        VerboseLevel.INFO: logging.INFO,
        VerboseLevel.DEBUG: logging.DEBUG
    }
    
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable logging completely if NONE
    if level == VerboseLevel.NONE:
        logger.disabled = True
    
    return logger

def set_logger_level(logger: logging.Logger, level: VerboseLevel):
    """
    Set the verbosity level of an existing logger.
    
    Args:
        logger: Logger object
        level: Verbosity level from VerboseLevel enum
    """
    logger.setLevel(level_mapping.get(level, logging.INFO))
    
    # Disable logging completely if NONE
    if level == VerboseLevel.NONE:
        logger.disabled = True
        
    return logger