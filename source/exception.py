import sys  # Used to access system-specific information, including exception details
from source.custom_logging import logging  # Assumes logging functionality is defined elsewhere for structured error logging

# Function to generate a detailed error message
def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message including the script name, line number, and error description.
    
    Parameters:
    - error: The actual error/exception object.
    - error_detail: The sys module to fetch exception details.
    
    Returns:
    - A formatted string with detailed error information.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extracts traceback information
    file_name = exc_tb.tb_frame.f_code.co_filename  # Retrieves the name of the file where the error occurred
    error_message = "The Error is taking place in python script name [{0}], line number [{1}], and the error message is [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Formats the file name, line number, and error message into a string
    )
    return error_message  # Returns the detailed error message

# Custom exception class that enhances default Python exceptions
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the custom exception with detailed error information.

        Parameters:
        - error_message: The original error message.
        - error_detail: The sys module to fetch detailed exception information.
        """
        super().__init__(error_message)  # Initialize the base Exception class with the original error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Generate a detailed error message

    def __str__(self):
        """
        Overrides the string representation of the exception to return the detailed error message.
        """
        return self.error_message  # Returns the custom error message when the exception is printed
