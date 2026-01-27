import pandas as pd
import numpy as np
import sys

from src.exception import MyException
from src.logger import logging


class PredictionTransformer:

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SAME transformations as training
        """

        try:
            logging.info("Starting prediction preprocessing")

            df = df.copy()

            # Drop Name
            if "Name" in df.columns:
                df.drop(columns=["Name"], inplace=True)

            # Combine Satisfaction
            if "Job Satisfaction" in df.columns and "Study Satisfaction" in df.columns:
                df["Satisfaction"] = df["Job Satisfaction"].combine_first(
                    df["Study Satisfaction"]
                )
                df.drop(["Job Satisfaction", "Study Satisfaction"], axis=1, inplace=True)

            # Combine Pressure
            if "Work Pressure" in df.columns and "Academic Pressure" in df.columns:
                df["Pressure"] = df["Work Pressure"].combine_first(
                    df["Academic Pressure"]
                )
                df.drop(["Work Pressure", "Academic Pressure"], axis=1, inplace=True)

            # CGPA handling
            df.loc[
                df["Working Professional or Student"] == "Working Professional",
                "CGPA"
            ] = df.loc[
                df["Working Professional or Student"] == "Working Professional",
                "CGPA"
            ].fillna(0)

            # Profession
            df["Profession"] = df.apply(
                lambda row: "Student"
                if row["Working Professional or Student"] == "Student"
                and pd.isnull(row["Profession"])
                else row["Profession"],
                axis=1,
            )

            df["Profession"] = df["Profession"].fillna("Teacher")

            # City
            df["City"] = df["City"].fillna("other")

            # Dietary
            diet_map = {
                "More Healthy": "Healthy",
                "No Healthy": "Unhealthy",
                "Less Healthy": "Unhealthy",
                "Less than Healthy": "Unhealthy"
            }

            df["Dietary Habits"] = df["Dietary Habits"].replace(diet_map)
            df["Dietary Habits"] = df["Dietary Habits"].fillna("Healthy")

            # Degree
            df["Degree"] = df["Degree"].fillna("B.Ed")

            # Sleep
            sleep_map = {
                'Less than 5 hours': 'Very Low Sleep',
                '7-8 hours': 'High Sleep',
                'More than 8 hours': 'Very High Sleep',
                '5-6 hours': 'Medium Sleep',
                '3-4 hours': 'Very Low Sleep',
                '6-7 hours': 'Medium Sleep',
                '8 hours': 'High Sleep',
                '8-9 hours': 'Very High Sleep',
                '9-11 hours': 'Very High Sleep',
                '10-11 hours': 'Very High Sleep',
                '9-5 hours': 'Medium Sleep'
            }

            df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)
            df["Sleep Duration"] = df["Sleep Duration"].fillna("Medium Sleep")

            # Numeric nulls
            num_cols = ["Financial Stress", "Satisfaction", "Pressure"]

            for col in num_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

            logging.info("Prediction preprocessing completed")

            return df

        except Exception as e:
            raise MyException(e, sys)
