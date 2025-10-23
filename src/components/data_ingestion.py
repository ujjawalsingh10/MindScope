import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.data_exporter import ProjData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
    
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        This method exports data from MOngoDB to csv file

        Output: data is returned as artifact of data ingestion components
        """
        try:
            logging.info(f"Exporting data from MongoDB")
            my_data = ProjData()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise MyException(e, sys)
    
    def split_data_as_train(self, dataframe: pd.DataFrame) -> None:
        """
        Split dataframe into train set and test set based on split ratio

        Output: Folder is created in s3 bucket
        """

        logging.info('Entered split_data_as_train_test method of Data_ingestion class')

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info('Performed train test split on the dataframe')
            logging.info('Exited split_data_as_train_test method of Data_Ingestion class')
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f'Exporting train and test file path')
            train_set = train_set.drop('_id', axis=1)
            # logging.info(f"Current columns in the Train Dataset: {train_set.columns}")
            test_set = test_set.drop('_id', axis=1)
            # logging.info(f"Current columns in the Test Dataset: {test_set.columns}")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path")
        
        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method initiates the data ingestion components of training pipeline
        Output: train set and test set are returned as the artifacts of data ingestion components
        """

        logging.info("Entered initiate_data_ingestion method of Data Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info('Got the data from mongoDB')

            self.split_data_as_train(dataframe)
            logging.info('Performd train test split on dataset')

            logging.info("Exited initiate_data_ingestion method of Data ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)

            logging.info(f'Data ingestion artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys)

