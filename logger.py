import sys
import logging
import structlog

from structlog.dev import ConsoleRenderer
from structlog.processors import TimeStamper, format_exc_info, add_log_level
from structlog.stdlib import LoggerFactory, BoundLogger

_RESET       = "\x1b[0m"
_BOLD        = "\x1b[1m"
_DIM         = "\x1b[2m"
_FG_GREEN    = "\x1b[32m"
_FG_CYAN     = "\x1b[36m"

def configure_structlog(debug: bool = False) -> None:

    app_log = logging.getLogger("supersonic")
    app_log.setLevel(logging.DEBUG if debug else logging.INFO)
    if not app_log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        app_log.addHandler(h)

    default_styles = ConsoleRenderer.get_default_level_styles(colors=True)
    custom_styles = {
        **default_styles,
        "debug": _DIM + _FG_CYAN,
        "info":  _BOLD + _FG_GREEN,
    }
    structlog.configure(
        processors=[
            add_log_level,                # adds the log level to each event
            TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            format_exc_info,              # format exception info if provided
            structlog.dev.ConsoleRenderer(pad_event=30, level_styles=custom_styles),
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> BoundLogger:
    return structlog.get_logger(f"supersonic.{name}")