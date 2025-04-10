import logging  # Python's built-in module for logging messages
import os  # Used for file and directory operations
from datetime import datetime  # Used to generate timestamps for log file names

# Generate a unique log file name with the current timestamp
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # E.g., "11_26_2024_10_30_00.log"

# Define the path for the logs directory and log file
log_path = os.path.join(os.getcwd(), "logs", log_file)  # Create a path like "<current_directory>/logs/<log_file>"
os.makedirs(log_path, exist_ok=True)  # Create the logs directory if it doesn't already exist

# Full path to the log file
path_log_file = os.path.join(log_path, log_file)

# Configure the logging system
logging.basicConfig(
    filename=path_log_file,  # Log messages will be written to this file
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Set the logging level to INFO (logs INFO and above: WARNING, ERROR, CRITICAL)
)

if __name__=="__main__":
    logging.info("logging")