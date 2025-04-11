## End to End Used Car Prediction Price with Deployment in Azure


## Link to the application:

https://end-to-end-used-car-prediction-website.onrender.com

## About the Project

The Project contains the following code files that will be shortly explained:

* setup.py
* .gitignore
* requirements.txt
* jupyter notebook-usedcarpriceprediction
* custom_logging.py
* exception.py
* utils.py
* data_loader.py
* data_processing.py
* model_builder.py
* prediction_pipeline.py
* home.html
* appy.py

## Resources Used

**Python Version**: 3.12.7

**Code Editor**: Visual Studio Code

**Packages**: numpy, pandas, matplotlib, seaborn, scikit-learn, ipykernel, catboost, xgboost, dill, flask, flask_cors, lime, os

**Dataset**: www.kaggle.com/datasets/ayaz11/used-car-price-prediction

**ProcessedDataset**: https://drive.google.com/file/d/1ViJxsGsbHPXD380_0GPYJbg1wuJoQOVJ/view?usp=drive_link

## Code Files

## setup.py

The script is designed to package the project for distribution, whether for upload to PyPI or internal use, install the project's dependencies listed in `requirements.txt`, and provide metadata that helps others understand and use the project. 

## .gitignore

It specifies files and directories Git should ignore to prevent them from being tracked in version control. The provided file excludes unnecessary files like compiled Python files (__pycache__/, *.pyc), build artifacts (build/, dist/), environment files (.env, venv/), logs, test coverage reports, and IDE or tool-specific files (e.g., .idea/, .ipynb_checkpoints).

## requirements.txt

It includes all the necessary libraries for the project. The main goal is to make installation smoother.

## UsedCarPricePrediction.ipynb

The main goal of this file was to:

1 - Clean and structure the data: duplicated values and outliers were eliminated, datatypes were corrected, columns were divided to create others that added more value to the analysis, column values were addapted.
2 - Various graphs were generated to gain insights and better understand the data.

## Customlogging.py

The code sets up a logging system that creates a uniquely named log file (based on the current timestamp) in a `logs` directory, formats log messages with details (timestamp, line number, log level, etc.), and logs messages at the INFO level or higher.

## exception.py

The code defines a custom exception handling system that generates detailed error messages, including the script name, line number, and the error description. It includes a function to format the error details and a custom exception class that enhances the default Python exceptions with more informative error messages.

## utils.py

The utils.py file contains utility functions that assist with common tasks throughout the project.It centralizes functionality that is used across different parts of the codebase, helping to keep the main code clean and organized. 

## data_loader.py

This script handles the process of loading and splitting data into training and testing sets. It first loads a dataset from a CSV file, saves the raw data, and then splits it into training and testing subsets (80-20 split). These subsets are saved to specified file paths, and the paths for the training and testing data are returned for further use.

## data_loader.py

This script is responsible for transforming and preprocessing data. It defines a configuration for saving the preprocessor object and implements a data transformation pipeline for both numerical and categorical columns. It handles imputing missing values, scaling data, and encoding categorical features. It also splits the data into features and target columns, applies the transformations, and saves the preprocessor object for future use.

## model_builder.py

This script trains different machine learning models on the processed data, using a variety of regression algorithms like Random Forest, Decision Tree, and XGBRegressor. It evaluates each model's performance, selects the best model based on its score, and saves this model to a file. The script also computes the R-squared score of the best model on the test set and returns this score.

## predictpipeline.py

It loads a pre-trained model and preprocessor from files, scales the input features, and returns predictions based on the processed data. The `CustomData` class stores user-provided car data (like year, miles, accidents, etc.) and converts it into a `pandas` DataFrame for prediction. Both classes include error handling with a custom exception (`CustomException`) to manage failures during their operations.

## home.html

It defines the FrontEnd using HTML. 

## app.py

This code defines a Flask web application with two routes: the home page (`/`) and a prediction page (`/predictdata`). The home page renders an HTML template (`index.html`), while the prediction page handles both GET and POST requests. For POST requests, it collects user input (e.g., car details), creates a `CustomData` object, and uses a `PredictPipeline` to generate predictions. The results are then displayed on the same page. The application runs in debug mode on any available IP address.


## Video Demonstration of the App on YouTube

To give you a quick overview of how the app works, I’ve created a demonstration video. While setting up the app locally can take a few minutes, this video allows you to experience the website without any setup.

[![Watch the Demo](https://img.youtube.com/vi/1nx-VaPryHE/0.jpg)](https://www.youtube.com/watch?v=1nx-VaPryHE)
