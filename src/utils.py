import os
import sys
import joblib
from src.exception import custom_exception

# saving object
def model_save(weights, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            joblib.dump(weights,f)
    
    except Exception as e:
        raise custom_exception(e,sys)