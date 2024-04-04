import os
import sys
import joblib
from src.logger import logging
from src.exception import custom_exception
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# saving object
def object_save(object, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            joblib.dump(object,f)
    
    except Exception as e:
        raise custom_exception(e,sys)
    

def model_evaluation(x_train, x_test, y_train, y_test, models, parameters):
    try:
        logging.info("Model_evaluation started ")
        detailed_report = {}
                
        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(model)
            parameter =  parameters[list(models.keys())[i]]
            grid_search = GridSearchCV(model, parameter, cv=5)
            grid_search.fit(x_train, y_train)
            
            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            print(f"Train score: {train_model_score}")
            print(f"Test score: {test_model_score}")
            detailed_report[list(models.keys())[i]] = test_model_score
        return detailed_report       
            
    except Exception as e:
        raise custom_exception(e,sys)
    
    

def object_load(file_path):
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise custom_exception(e,sys)
        
    
