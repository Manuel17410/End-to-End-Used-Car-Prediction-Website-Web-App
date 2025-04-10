import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# root directory of your project to sys.path to make imports easier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 

from source.exception import CustomException

# save_object function is responsible for serializing (saving) Python objects into a file.
def save_object(file_path, obj):
    try:
        # Get the directory path where the object will be saved
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

       # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        # Initialize a report dictionary to store model evaluation scores
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)