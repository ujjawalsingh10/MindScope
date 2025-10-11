import sys
import logging

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self._get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)
        logging.error(self.error_message)  # optional auto-logging

    @staticmethod
    def _get_detailed_error_message(message, error_detail):
        _, _, exc_tb = sys.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown File"
            line_number = "Unknown Line"
        return f"[Error] {message} | Details: {error_detail} | File: {file_name} | Line: {line_number}"

    def __str__(self):
        return self.error_message
