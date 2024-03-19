import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
log_path = os.path.join(os.getcwd(), 
                        "logs")
print(log_path)
os.makedirs(log_path,exist_ok=True)

log_file_path = os.path.join(log_path, LOG_FILE)

logging.basicConfig(filename=log_file_path,
                    format = '[%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s]',
                    level=logging.INFO)



if __name__ == "__main__":
    logging.info("This is a test log message")
    logging.warning("This is a warning log message")
    logging.error("This is an error log message")

    