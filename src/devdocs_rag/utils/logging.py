import logging
import sys

from devdocs_rag import config


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stderr (stdout is reserved for MCP protocol)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    return logger
