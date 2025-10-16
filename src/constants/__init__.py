import os
from datetime import date

## For MONGODB connection
DATABASE_NAME = "mental_health_mlops"
COLLECTION_NAME = "raw_data"
MONGODB_URL_KEY = 'MONGO_DB_URL'

PIPELINE_NAME: str = ''
ARTIFACT_DIR: str = 'artifact'

MODEL_FILE_NAME = 'model.pkl'

TARGET_COLUMN = 'Response'
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'

FILE_NAME: str = 'data.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')

"""
Data Ingestion related cosntants
"""
DATA_INGESTION_COLLECTION_NAME: str = 'raw_data'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20

"""
Data Validation related constant
"""
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_REPORT_FILE_NAME: str = 'report.yaml'
