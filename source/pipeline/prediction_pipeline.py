import sys
import os
import pandas as pd
from source.exception import CustomException
from source.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        year: int,
        miles: int,
        accidents: int,
        number_of_owners: int,
        brand: str,
        color_exterior: str,
        color_interior: str):

        self.year = year

        self.miles = miles

        self.accidents = accidents

        self.number_of_owners = number_of_owners

        self.brand = brand

        self.color_exterior = color_exterior

        self.color_interior= color_interior

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "year": [self.year],
                "miles": [self.miles],
                "accidents": [self.accidents],
                "number_of_owners": [self.number_of_owners],
                "brand": [self.brand],
                "color_exterior": [self.color_exterior],
                "color_interior": [self.color_interior],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)