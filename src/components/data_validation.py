import json
import sys
import os
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_numer_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        This method validates the number of columns
        Output: Returns bool value based on validation results
        """
        try:
            status = len(dataframe.columns) == len(self.schema_config['columns'])
            logging.info(f"Required number of columns present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)
    
    def is_column_exist(self, df: pd.DataFrame) -> bool:
        """
        This method validates the existence of numerical and categorical columns
        Output: Return bool value based on validation results
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            # check for missing numerical columns
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")
            
            # check for missing categorical columns
            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")
            
            return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
        
        except Exception as e:
            raise MyException(e, sys) from e
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation component for the pipeline
        Output: Returns bool value based on validation results
        """

        try:
            validation_error_msg = ''
            logging.info('Starting data validation')
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            ## check col len of DataFrame for train/test df
            status = self.validate_numer_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f'Columns are missing in training dataframe'
            else:
                logging.info(f'All required columns present in training dataframe: {status}')
            
            status = self.validate_numer_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns not present in testing dataframe: {status}"
            else:
                logging.info(f"All required columns present in testing dataframe")
            
            # Validating col dtype for train/test df
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")
            
            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            ## ensure the directory for validation_report_file_path_exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            ## save validation status and message to json file
            validation_report = {
                'validation_status': validation_status,
                'message': validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, 'w') as f:
                json.dump(validation_report, f, indent=4)
            
            logging.info('Data validation artifact created and saved to JSON file.')
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e