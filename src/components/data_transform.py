import os
import sys
import numpy as np
import pandas as pd 
from src.logger import logging
from src.utils import object_save
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import custom_exception
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("models","preprocessor.pkl")



class DataTransform:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()
        
    
    def get_data_to_transform(self):
        logging.info("Data transformation started ")
        try :
            numarical_features = ["reading_score", "writing_score"]
            catagorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            numarical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            catagorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)) 
                    
                ]
            )
        
            logging.warning("with_mean = FALSE for sparce metrics memory limitation error")
            
            logging.info("created numarical_column: ".format(numarical_features) )
            logging.info("created catagorical_column: ".format(catagorical_features) )
            
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numarical_pipeline,numarical_features),
                    ("cat",catagorical_pipeline,catagorical_features)
                ],
                remainder='drop'
            )
            return preprocessor
        
        except Exception as e:
            raise custom_exception(e,sys)
        
    
    
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            logging.info("Sucessfully read train data")
            
            test_df = pd.read_csv(test_data_path)
            logging.info("Sucessfully read test data")
            
            # preprocess
            preprocessor_object = self.get_data_to_transform()
            logging.info("Sucessfully obtained preprocessor object")
            
            # spliting targert feature and input features from train and test dataframe
            target_feature_name = "math_score"
            
            # Explicit data splitting
            X_train, X_test, y_train, y_test = train_test_split(
                train_df.drop(columns=[target_feature_name], axis=1),
                train_df[target_feature_name],
                test_size=0.3,
                random_state=42,
            )
            
            # Preprocessing
            preprocessor_object = self.get_data_to_transform()
            logging.info("Sucessfull split target_feature from test dataframe ")
            
            train_arr = preprocessor_object.fit_transform(X_train, y_train)
            
            print(f"train_arr: {train_arr.shape}")
            test_arr = preprocessor_object.transform(X_test)
            print(f"test_arr: {test_arr.shape}")
            
            
            
            logging.info("Sucessfull fit transform train data and transfom test data")
            
            
            # Saving the object
            object_save(preprocessor_object,self.data_transformation.preprocessor_path)
            
            logging.info(f"Saved preprocessing object.")
            
            return train_arr, test_arr, self.data_transformation.preprocessor_path
            
            
        except Exception as e:
            raise custom_exception(e,sys)
        