import sys
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes: int = 1
        self.no: int = 0
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing object: Input Object of preprocessor
        :param trained_model object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
    
    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info('Starting prediction process...')

            # Step 1: Apply scaling transformations using pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info('Using the trained model to get predictions')
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions
        except Exception as e:
            logging.error('Error occurred in predict method', exc_info=True)
            raise MyException(e, sys) from e
        
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"