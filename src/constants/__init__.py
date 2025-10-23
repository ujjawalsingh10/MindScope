import os
from datetime import date

## For MONGODB connection
DATABASE_NAME = "mental_health_mlops"
COLLECTION_NAME = "raw_data"
MONGODB_URL_KEY = 'MONGO_DB_URL'

PIPELINE_NAME: str = ''
ARTIFACT_DIR: str = 'artifact'

MODEL_FILE_NAME = 'model.pkl'

TARGET_COLUMN = 'Depression'
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

"""
Data Transformation related constants
"""
DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = 'transformed_object'

"""
MODEL TRAINER related constants
"""
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR: str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME: str = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join('config', 'model.yaml')
MODEL_TRAINER_N_ESTIMATORS = 100
MODEL_TRAINER_LEARNING_RATE: float = 0.1

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = 'model-mindscope-mlops'
MODEL_PUSHER_S3_KEY = 'model-registry'

APP_HOST = '0.0.0.0'
APP_PORT = 5000

