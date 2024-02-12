import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs", log_file_name=None):
    # Create the directory for logs if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # If log_file_name is None, generate it with the current date and time
    if log_file_name is None:
        log_file_name = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
    
    # Full path for the log file
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # Set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path)
        ]
    )

