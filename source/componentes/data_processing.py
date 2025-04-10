# Importing necessary libraries for data transformation and preprocessing

from sklearn.compose import ColumnTransformer # Used to apply different transformations to different subsets of features
from sklearn.impute import SimpleImputer # For filling missing values in the dataset
from sklearn.pipeline import Pipeline # To streamline the process of applying transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler # For encoding categorical data and scaling numerical data
import numpy as np
import pandas as pd
import os
import sys
from source.exception import CustomException
from source.custom_logging import logging
from source.utils import save_object  # Function to save objects (like transformers) to disk
from dataclasses import dataclass  # For creating configuration classes with default values

# Data class to store configuration settings for data processing, such as the path for saving the preprocessor object
@dataclass
class DataProcessingConfiguration:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl") # Path for saving the preprocessing object

# DataProcessing class that handles all transformations on the dataset
class DataProcessing:
    def __init__(self):
        # Initialize the configuration for data transformation (preprocessor object path)
        self.data_transformation_config = DataProcessingConfiguration()

    # Method to create and return the data transformer (preprocessor) object
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["year", "miles", "accidents", "number_of_owners"]
            categorical_columns = ["brand", "color_exterior", "color_interior"]

            # Define a pipeline for numerical columns:
            # - Impute missing values using median
            # - Scale the data using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Define a pipeline for categorical columns:
            # - Impute missing values using the most frequent value
            # - OneHotEncode the categorical data
            # - Scale the encoded features using StandardScaler
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns), # Apply numerical pipeline on numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns) # Apply categorical pipeline on categorical columns
                ]
            )

            # Return the combined preprocessor object
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Method to initiate data processing on the train and test datasets
    def initiate_DataProcessing(self, train_path, test_path):
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Successfully loaded train and test datasets.")

            # Get the preprocessor object that handles transformations
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"
            numerical_columns = ["year", "miles", "accidents", "number_of_owners"]

            # Separate features and target for training
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features and target for testing
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations on datasets.")

            # Transform features using the preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse arrays to dense arrays 
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Ensure targets are reshaped for concatenation
            target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)

            # Concatenate features and targets
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_df])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_df])

            logging.info("Transformation complete. Saving preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)









        
