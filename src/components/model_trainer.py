import os
import sys
import numpy as np
from src.logger import logging
from xgboost import XGBRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from src.exception import custom_exception
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from src.utils import  object_save,model_evaluation
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import Lasso, Ridge


@dataclass
class ModelTrainConfig:
    final_model_path = os.path.join("models","trained_model.pkl")
    lambda_param: float = 0.1


class ModelTrain:
    def __init__(self) -> None:
        self.model_train_config = ModelTrainConfig()
        
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting data into train and test data input")
            
            x_train, x_test, y_train, y_test = (
                train_array[:,:-1], # x_train dependent variables
                test_array[:,:-1],  #x_test dependent variables
                train_array[:,-1], #y_train independent variable
                test_array[:,-1] #y_test independent variable
            )
            
            models = {
                "decision_tree":DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor(),
                "knn": KNeighborsRegressor(),
                "cat_boost": CatBoostRegressor(logging_level='Silent'),
                "xgboost": XGBRegressor(),
                "lasso_regression": Lasso(alpha=self.model_train_config.lambda_param),  
                "ridge_regression": Ridge(alpha=self.model_train_config.lambda_param) 
            }
            
            parameters = {
                "decision_tree": {
                    "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter" : ['best','random'],
                    "max_features" : ['sqrt','log2']
                    },
                
                "random_forest": {
                    "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "n_estimators" : [80,100, 150, 200, 250],
                    "max_features" : ['sqrt','log2']
                    },
                
                "knn": {
                    "n_neighbors" : [5,10, 20 ],
                    "weights" : ['uniform','distance'],
                    "algorithm" : ['auto','ball_tree', 'kd_tree', 'brute'],
                    },
                
                "cat_boost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                    },
                
                "xgboost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [80,100, 150, 200, 250]
                    },
                "lasso_regression": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
                    },
                
                "ridge_regression": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
                    }
            }
            
            logging.info("Training models")
            
            model_report:dict = model_evaluation(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test , models = models, parameters = parameters)
            logging.warning(f"Model report: {model_report}")
            best_model_name = max(model_report, key=lambda model: model_report[model]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            
            best_model = models[best_model_name]
            
            
            if best_model_score < 0.65:
                raise custom_exception("Model score is less than 0.65",sys)
            
            logging.info(f"Best model is {best_model_name} with score {best_model_score}")            
            
            object_save(object=best_model, file_path=self.model_train_config.final_model_path)
            logging.info(f"Model saved at {self.model_train_config.final_model_path}")
            prediction = best_model.predict(x_test)
            r2 = r2_score(y_test, prediction)
            
            return r2
          
        except Exception as e:
            raise custom_exception(e,sys)