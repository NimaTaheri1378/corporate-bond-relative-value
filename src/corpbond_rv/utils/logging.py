from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure console/file logging with a compact research-pipeline format."""
    logger = logging.getLogger("corpbond_rv")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
