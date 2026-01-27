import sys
from src.entity.config_entity import DepressionPredictorConfig
from src.pipeline.prediction_transformer import PredictionTransformer
from src.entity.s3_estimator import ProjEstimator
from src.exception import MyException
from src.logger import logging
import pandas as pd

class MentalHealthData:
    """
    This class collects RAW user input
    (same format as original dataset)
    """
    def __init__(
        self,
        Name,
        Gender,
        Age,
        City,
        Working_Professional_or_Student,
        Profession,
        Academic_Pressure,
        Work_Pressure,
        CGPA,
        Study_Satisfaction,
        Job_Satisfaction,
        Sleep_Duration,
        Dietary_Habits,
        Degree,
        Suicidal_Thoughts,
        Work_Study_Hours,
        Financial_Stress,
        Family_History
    ):
        try:
            self.Name = Name
            self.Gender = Gender
            self.Age = Age
            self.City = City
            self.Working_Professional_or_Student = Working_Professional_or_Student
            self.Profession = Profession
            self.Academic_Pressure = Academic_Pressure
            self.Work_Pressure = Work_Pressure
            self.CGPA = CGPA
            self.Study_Satisfaction = Study_Satisfaction
            self.Job_Satisfaction = Job_Satisfaction
            self.Sleep_Duration = Sleep_Duration
            self.Dietary_Habits = Dietary_Habits
            self.Degree = Degree
            self.Suicidal_Thoughts = Suicidal_Thoughts
            self.Work_Study_Hours = Work_Study_Hours
            self.Financial_Stress = Financial_Stress
            self.Family_History = Family_History

        except Exception as e:
            raise MyException(e, sys)

    def get_input_dataframe(self) -> pd.DataFrame:
        """Convert input to DataFrame"""

        try:
            data = {
                "Name": [self.Name],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "City": [self.City],
                "Working Professional or Student": [self.Working_Professional_or_Student],
                "Profession": [self.Profession],
                "Academic Pressure": [self.Academic_Pressure],
                "Work Pressure": [self.Work_Pressure],
                "CGPA": [self.CGPA],
                "Study Satisfaction": [self.Study_Satisfaction],
                "Job Satisfaction": [self.Job_Satisfaction],
                "Sleep Duration": [self.Sleep_Duration],
                "Dietary Habits": [self.Dietary_Habits],
                "Degree": [self.Degree],
                "Have you ever had suicidal thoughts ?": [self.Suicidal_Thoughts],
                "Work/Study Hours": [self.Work_Study_Hours],
                "Financial Stress": [self.Financial_Stress],
                "Family History of Mental Illness": [self.Family_History]
            }

            df = pd.DataFrame(data)

            logging.info("Created user input DataFrame")

            return df

        except Exception as e:
            raise MyException(e, sys)        

class MentalHealthPredictor:
    def __init__(self, prediction_pipeline_config: DepressionPredictorConfig = DepressionPredictorConfig()) -> None:
        """
        param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.transformer = PredictionTransformer()
        except Exception as e:
            raise MyException(e, sys)
    
    def predict(self, dataframe) -> str:
        """
        This is a method for MentalHeathPredictor
        
        :return: Prediction for the user input
        """
        try:
            logging.info('Entered predict method of MentalHeathPredictor class')

            logging.info('Applying transformations to the input data')
            df_processed = self.transformer.transform(dataframe)
            logging.info('Transformations done !!')

            model = ProjEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            result = model.predict(df_processed)
            return result
        except Exception as e:
            raise MyException(e, sys)