import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(log_dir, log_file)

# Create a custom logger
logger = logging.getLogger("ml_pipeline_logger")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    #Without the console handler → all logs go silently to the .log file.
    #With the console handler → you also see key log messages in real-time in your terminal, while they’re being written to the file.
    # Attach console handler to the same custom logger
    # Creates a handler that outputs to the console (stdout).
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(asctime)s] %(message)s")
    # Controls how messages appear in the console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
