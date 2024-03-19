import sys
from src.logger import logging



def error_message_details(error,error_details:sys):
    a,b,exc_tb = error_details.exc_info()
    print(a)
    print(b)
    print(exc_tb)
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message


class custom_exception(Exception):
    def __init__(self, error, error_details:sys):
        self.error = error
        self.error_details = error_details
        self.error_message = error_message_details(self.error, self.error_details)
        logging.error(self.error_message)
        super().__init__(self.error_message)
        self.error_details = error_details

    def __str__(self):
        return self.error_message