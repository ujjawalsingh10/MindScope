import os
import sys
from src.logger import configure_logger
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data):
        