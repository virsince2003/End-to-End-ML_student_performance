import os
import sys
import pandas as pd 
from src.logger import logging
from src.utils import model_save
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import custom_exception
from sklearn.compose import ColumnTransformer
from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("models","preprocessor.pkl")



class DataTransformation:
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
                    ("imputer", SimpleImputer(strategy="most-frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
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
                ]
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
            
            # train dataframe
            input_features_train_df = train_df.drop(columns=[target_feature_name],axis=1)
            target_feature_train_df = train_df[target_feature_name]
            logging.info("Sucessfull split target_feature from train dataframe ")
            
            #test dataframe
            input_features_test_df = train_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df = train_df[target_feature_name]
            logging.info("Sucessfull split target_feature from test dataframe ")

            
            train_arr = preprocessor_object.fit_transform(input_features_train_df,target_feature_train_df)
            test_arr = preprocessor_object.transform(input_features_test_df,target_feature_test_df)
            
            logging.info("Sucessfull fit transform train data and transfom test data")
            
            # Saving the object
            model_save(preprocessor_object,self.data_transformation.preprocessor_path)
            
            logging.info(f"Saved preprocessing object.")
            
            return train_arr, test_arr, self.data_transformation.preprocessor_path
            
            
        except Exception as e:
            raise custom_exception(e,sys)
        