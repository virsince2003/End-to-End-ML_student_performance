import os
import sys
import pandas as pd
from src.logger import logging
from src.utils import object_load 
from dataclasses import dataclass
from src.exception import custom_exception


@dataclass
class model_and_preprocessor_path:
    model_path: str = os.path.join("models", "trained_model.pkl")
    preprocessor_path:str = os.path.join("models","preprocessor.pkl")


class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education :str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int
                ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def to_dataframe(self):
        try:
            custom_data_dict = {
                "gender" : [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_dict)
            
        except Exception as e:
            raise custom_exception(e,sys)
        



class PredictionPipeline:
    
    def __init__(self):
        self.model_and_preprocessor_path = model_and_preprocessor_path()
        
    def prediction(self,features):
        logging.info("Initiating the prediction pipe")
        try:
            model = object_load(file_path=self.model_and_preprocessor_path.model_path)
            logging.info("Model loaded successfully")
            
            preprocessor = object_load(file_path=self.model_and_preprocessor_path.preprocessor_path)            
            logging.info("Preprocessor loaded successfully")
            print(type(features))
            print(features)
            scaled_features = preprocessor.transform(features)
            scaled_features = scaled_features[:, :18]
            logging.info("Features scaled successfully")
            
            prediction = model.predict(scaled_features)
            return prediction[0]        
        
        except Exception as e:
            raise custom_exception(e,sys)
