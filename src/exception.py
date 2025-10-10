import sys
# Whenever we get a error we want to know the file name and line number where the error has occurred. This is done using sys module. I am going to get my custom error message using error_message_detail function and pass that message to CustomException class.
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {exc_tb.tb_lineno} error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        #initialize the parent Exception class with this error message
        super().__init__(error_message)
        #instead of just storing the raw error_message, we call a helper function error_message_detail(). This function will give us a detailed error message including the file name and line number where the error occurred.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message