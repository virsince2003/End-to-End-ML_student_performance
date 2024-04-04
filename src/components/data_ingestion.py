import os
import sys
import pandas as pd
# sys.path.append("E:\End-to-End_ML_Projects\End-to-End-ML_student_performance")
from src.logger import logging
from dataclasses import dataclass
from src.exception import custom_exception
# from src.components.model_trainer import ModelTrain
from sklearn.model_selection import train_test_split
# from src.components.data_transform import DataTransform, DataTransformationConfig

"""
Gathering data from the source

"""

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    data_path : str = os.path.join("data","student_data.csv")
    

class DataIngestion:
    def  __init__(self):
        self.data_ingestion = DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("Iniating the csv reading process")
        try:
            """
            readin the student data csv file and spliting it to train and split
            """
            df = pd.read_csv(self.data_ingestion.data_path)
            logging.info("reading csv sucessfully")
            
            train_data,test_data = train_test_split(df, test_size=0.3, random_state=42)
            logging.info("splitting sucessfull train and test")
            
            # train file
            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path), exist_ok=True)
            
         # pushing  train and test data to csv
            train_data.to_csv(self.data_ingestion.train_data_path,index=False, header=True)
            
            # test file making
            os.makedirs(os.path.dirname(self.data_ingestion.test_data_path), exist_ok=True)
            test_data.to_csv(self.data_ingestion.test_data_path,index=False, header=True)
            
            logging.info("Data Ingestion completed")
            
            return self.data_ingestion.train_data_path, self.data_ingestion.test_data_path
        
        except Exception as e:
            raise custom_exception(e,sys)
