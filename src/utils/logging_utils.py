"""Logging utilities for Foodwarm."""
import logging
import sys
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(name: str = "foodwarm", log_dir: str = "./outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """Setup rich console logger and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s", datefmt="[%X]")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/train.log")
    file_handler.setLevel(level)
    file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
