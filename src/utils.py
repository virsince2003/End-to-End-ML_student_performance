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
        logging.info("Model evaluation started")
        detailed_report = {}
        lambda_param  = [0.001, 0.01, 0.1, 1.0]

        for model_name, model in models.items():
            # Access parameters using model name
            param_grid = parameters[model_name]

            if model_name in ["lasso_regression", "ridge_regression"]:
                param_grid["alpha"] = lambda_param
        
            # Perform GridSearchCV with cross-validation
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(x_train, y_train)

            # Get best parameters
            best_params = grid_search.best_params_

            # Update model with best parameters
            model.set_params(**best_params)

            # Fit the model with optimized parameters
            model.fit(x_train, y_train)

            # Make predictions on train and test sets
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculate scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Store results in detailed report
            detailed_report[model_name] = {
                "train_score": train_score,
                "test_score": test_score,
                "best_params": best_params,
            }

        return detailed_report

    except Exception as e:
        raise custom_exception(e, sys)

    
    

def object_load(file_path):
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise custom_exception(e,sys)
        
    
