import structlog
import sys
import logging
import time
from typing import Any, Dict
from structlog.dev import ConsoleRenderer
from structlog.processors import TimeStamper

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Configure which loggers to show
LOGGER_LEVELS = {
    "supersonic.app": logging.INFO,
    "supersonic.server": logging.INFO,
    "supersonic.service": logging.INFO,
    "supersonic.main": logging.INFO,
}

class LoggerFilter(logging.Filter):
    def filter(self, record):
        # Get the logger name without the 'supersonic.' prefix
        logger_name = record.name.replace('supersonic.', '')
        
        # Check if we have a specific level for this logger
        if record.name in LOGGER_LEVELS:
            return record.levelno >= LOGGER_LEVELS[record.name]
        
        # Default to INFO level for unknown loggers
        return record.levelno >= DEFAULT_LOG_LEVEL

# First, configure the standard library logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=DEFAULT_LOG_LEVEL,
)

# Add our filter
logging.getLogger().addFilter(LoggerFilter())

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True, pad_event=25),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(DEFAULT_LOG_LEVEL),
    cache_logger_on_first_use=True,
)

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the specified name.
    Returns a bound logger that can be used for structured logging.
    
    Example usage:
        logger = get_logger("app")
        logger.info("event_name", 
                   model_name="model1", 
                   status="success",
                   extra_field="value")
    """
    return structlog.get_logger(f"supersonic.{name}")

# Create a default logger for direct use
logger = get_logger("root") 