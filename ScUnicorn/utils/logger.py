# utils/logger.py
"""
Logger Utility for ScUnicorn
Provides a reusable logger to track training progress and evaluation results.
"""
import logging
from datetime import datetime
import os

def get_logger(log_dir="logs", log_name="scunicorn_log"):
    """
    Create and configure a logger for ScUnicorn.

    Parameters:
    - log_dir (str): Directory to save log files.
    - log_name (str): Name of the log file (without extension).

    Returns:
    - logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if logger is reused
    if logger.hasHandlers():
        return logger

    # Create file handler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    # Test the logger utility
    print("Testing logger utility...")

    # Get logger instance
    logger = get_logger(log_dir="logs", log_name="test_log")

    # Log messages
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print("Logger utility test passed.")
