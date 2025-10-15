import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class ProjData:
    """
    class to export the project data from mongoDB
    """
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)
    
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports entire database collection as Dataframe
        """
        try:
            ## access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

        
            ### convert collection data to dataframe and preprocess
            print(f"Fetching data from mongoDB....")
            df = pd.DataFrame(list(collection.find()))
            print(f"Fetched Data with {len(df)}")
            return df
        except Exception as e:
            raise MyException(e, sys)
            
        