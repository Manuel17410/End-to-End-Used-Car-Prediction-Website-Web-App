# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import os  # For interacting with the operating system (e.g., paths)
import sys  # For system-specific parameters and functions
from source.exception import CustomException  # Custom exception class for error handling
from source.custom_logging import logging  # Custom logging for tracking events
from sklearn.model_selection import train_test_split  # For splitting the data into training and test sets
from dataclasses import dataclass  # For creating configuration classes with default values
from source.componentes.data_processing import DataProcessing  # Data processing component
from source.componentes.model_builder import ModelBuilderConfiguration  # Configuration for model building
from source.componentes.model_builder import ModelBuilder  # Model training component

# Define a dataclass for data loader configuration with default paths for data files
@dataclass
class DataLoaderConfiguration:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save the training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save the testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to save the raw data

# Class responsible for data loading and splitting
class DataLoader:
    def __init__(self):
        # Initialize configuration for data paths
        self.ingestion_config = DataLoaderConfiguration()

    def initiate_DataLoader(self):
        logging.info("Starting Data Loading process.")  # Log the beginning of the data loading process
        try:
            # Specify the file path for the dataset
            file_path = r'C:\Users\manue\Desktop\DataScience\Datasets\car_sales_preprocessed.csv'

            # Load the dataset into a Pandas DataFrame
            df = pd.read_csv(file_path)
            logging.info('Dataset loaded into DataFrame.')

            # Create the directory for artifacts if it doesn't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data in the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts.")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Save the training data to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing data to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test data saved to artifacts.")

            # Return the paths of the training and testing data files
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Instantiate the DataLoader class
    obj = DataLoader()

    # Initiate the data loading process and retrieve paths for training and testing data
    train_data, test_data = obj.initiate_DataLoader()

    # Instantiate the data processing component
    data_processing = DataProcessing()

    # Process the data and get transformed arrays along with the preprocessor's path
    train_arr, test_arr, preprocessor_path = data_processing.initiate_DataProcessing(train_data, test_data)

    logging.info(f"Data processing completed. Preprocessor saved at: {preprocessor_path}")

    # Instantiate the model trainer class
    modeltrainer = ModelBuilder()

    # Train the model and print the results
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))


    


    

