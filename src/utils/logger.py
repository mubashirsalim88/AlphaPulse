import logging
import os

def setup_logger(name, log_file):
    """Set up a logger with file output."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(logging.INFO)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger