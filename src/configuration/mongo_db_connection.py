import os
import sys
import pymongo
import certifi
from src.exception import MyException
from src.logger import logging
from dotenv import load_dotenv
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

load_dotenv()
## loading certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    client = None
    def __init__(self, database_name: str = DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set")
                
                ## establish connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name] ## connects to specified database
            self.database_name = database_name ## other wise

            logging.info('MongoDB connection successful')
        
        except Exception as e:
            raise MyException(e, sys)


        