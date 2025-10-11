import os
import sys
import pymongo
import certifi
from src.exception import CustomException
from src.logger import configure_logger
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

## loading certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    