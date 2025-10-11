import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

## log file constraints
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024 # 5 MB
BACKUP_COUNT = 3 ## no of backup log files to keep

## log file path construction

log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    ## defining formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    ## file handler with rotation
    file_handler = RotatingFileHandler(log_file_path, 
                                       maxBytes=MAX_LOG_SIZE,
                                       backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    ## adding handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
