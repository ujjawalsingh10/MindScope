import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.utils.main_utils import read_yaml_file, save_object, save_numpy_array
from src.exception import MyException
from src.logger import logging

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
    
    def get_data_transformer_object(self, X_train) -> Pipeline:
        """
        Creates and returns a data transformer object for the data
        """
        logging.info('Entered get_data_transformer_object method of DataTransformation class')

        try:
            # initialize transformers

            #load schema configurations
            num_features = X_train.select_dtypes(include=np.number).columns.tolist()
            cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
            
            # numeric transformer
            numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
            numeric_transformer = Pipeline(steps=numeric_steps)

            # categorical transformer
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ], remainder='drop', sparse_threshold=0)

            logging.info(f'Preprocessor created. Numeric cols: {num_features} | Categorical cols: {cat_features}')
            return preprocessor
        
        except Exception as e:
            raise MyException(e, sys)
    
    def _drop_id_column(self, df):
        """ Drop the id column if it exists"""
        logging.info('Dropping "id" column')
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis = 1)
        return df

    def _drop_duplicates(self, df):
        """Drops duplicate rows from the dataset"""
        total_dups = df.duplicated().sum()
        if total_dups > 0:
            logging.info(f"Dropping {total_dups} duplicate rows from dataset.")
            df = df.drop_duplicates()
        else:
            logging.info("No duplicate rows found in dataset.")
        return df
    
    def _combine_job_study_satisfaction(self, df):
        """
        Combine job satisfaction and study satisfaction into one 'Satisfaction' column.
        Drops the original Job and Study Satisfaction columns.
        """
        logging.info('Combining Job and Study Satisfaction columns')
        
        if 'Job Satisfaction' in df.columns and 'Study Satisfaction' in df.columns:
            df['Satisfaction'] = df['Job Satisfaction'].combine_first(df['Study Satisfaction'])
            df = df.drop(['Job Satisfaction', 'Study Satisfaction'], axis=1)
            logging.info('Successfully combined Job and Study Satisfaction into Satisfaction column')
        else:
            logging.warning('Job/Study Satisfaction columns not found in dataset; skipping merge')
        
        return df
    
    def _combine_work_academic_pressure(self, df):
        """Combine Work Pressure and Academic Pressure into Pressure"""
        if 'Work Pressure' in df.columns and 'Academic Pressure' in df.columns:
            df['Pressure'] = df['Work Pressure'].combine_first(df['Academic Pressure'])
            df.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)
            logging.info("Combined Work & Academic Pressure → Pressure")
        return df
    
    def _fill_cgpa_values(self, df, student_mean=None):
        """
        Fill CGPA:
        - Working Professionals → 0
        - Students → mean of Student CGPA (calculated on train)
        """
        if student_mean is None:
            student_mean = df.loc[df['Working Professional or Student'] == 'Student', 'CGPA'].mean()

        df.loc[df['Working Professional or Student'] == 'Working Professional', 'CGPA'] = (
            df.loc[df['Working Professional or Student'] == 'Working Professional', 'CGPA'].fillna(0)
        )

        df.loc[df['Working Professional or Student'] == 'Student', 'CGPA'] = (
            df.loc[df['Working Professional or Student'] == 'Student', 'CGPA'].fillna(student_mean)
        )
        logging.info("Filled CGPA for Professionals (0) and Students (mean)")
        return df, student_mean

    def _fill_profession(self, df, reference_df=None):
        """Fill profession based on rules"""
        df['Profession'] = df.apply(
            lambda row: 'Student' if row['Working Professional or Student'] == 'Student' and pd.isnull(row['Profession'])
            else row['Profession'], axis=1
        )
        if reference_df is None:
            reference_df = df
        df['Profession'] = df['Profession'].fillna('Teacher')
        logging.info("Filled Profession (Student/Teacher/mode)")
        return df
    
    def _map_city_values(self, df, reference_df):
        value_count = reference_df["City"].value_counts()
        df['City'] = df['City'].map(lambda x: x if value_count.get(x, 0) >= 10 else 'other')
        logging.info("Mapped rare City values → 'other'")
        return df
    
    def _map_dietary_habits(self, df, reference_df):
        specific_mappings = {
            'More Healthy': 'Healthy',
            'No Healthy': 'Unhealthy',
            'Less Healthy': 'Unhealthy',
            'Less than Healthy': 'Unhealthy'
        }
        value_count = reference_df['Dietary Habits'].value_counts()
        mode_value = reference_df['Dietary Habits'].mode()[0]

        def map_habits(v):
            if v in specific_mappings:
                return specific_mappings[v]
            elif value_count.get(v, 0) <= 2:
                return mode_value
            return v

        df['Dietary Habits'] = df['Dietary Habits'].map(map_habits)
        df['Dietary Habits'] = df['Dietary Habits'].fillna(mode_value)
        logging.info("Cleaned & mapped Dietary Habits")
        return df

    def _map_degree_values(self, df, reference_df):
        df['Degree'] = df['Degree'].fillna('B.Ed')
        value_count = reference_df["Degree"].value_counts()
        df['Degree'] = df['Degree'].map(lambda x: x if value_count.get(x, 0) > 4 else 'other')
        logging.info("Filled/mapped Degree values")
        return df

    def _map_sleep_duration(self, df, reference_df):
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
        df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)
        mode_value = reference_df['Sleep Duration'].mode()[0]
        df['Sleep Duration'] = df['Sleep Duration'].fillna(mode_value)
        logging.info("Mapped & filled Sleep Duration")
        return df

    def _fill_numerical_nulls(self, df, reference_df=None):
        """Fill remaining numerical nulls using mean (based on train reference)"""
        if reference_df is None:
            reference_df = df
        num_cols = ['Financial Stress', 'Satisfaction', 'Pressure']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(reference_df[col].mean())
        logging.info("Filled numerical nulls using mean values")
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline
        """
        try:
            logging.info('Data Transformation Started !!')
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info('Train Test data loaded')

            # Apply custom transformations in specified sequence
            
            # Step 0: Drop id column
            train_df = self._drop_id_column(train_df)
            test_df = self._drop_id_column(test_df)

             # Step 1: Drop duplicates
            train_df = self._drop_duplicates(train_df)
            test_df = self._drop_duplicates(test_df)

            # Step 2: Combine satisfaction and pressure columns
            train_df = self._combine_job_study_satisfaction(train_df)
            test_df = self._combine_job_study_satisfaction(test_df)
            train_df = self._combine_work_academic_pressure(train_df)
            test_df = self._combine_work_academic_pressure(test_df)

            # Step 3: Handle CGPA
            train_df, student_mean = self._fill_cgpa_values(train_df)
            test_df, _ = self._fill_cgpa_values(test_df, student_mean)

            # Step 4: Fill nulls (numeric + categorical)
            train_df = self._fill_numerical_nulls(train_df)
            test_df = self._fill_numerical_nulls(test_df, reference_df=train_df)

            # Step 5: Handle profession, city, dietary habits, degree, sleep
            train_df = self._fill_profession(train_df)
            test_df = self._fill_profession(test_df, reference_df=train_df)

            train_df = self._map_city_values(train_df, train_df)
            test_df = self._map_city_values(test_df, train_df)

            train_df = self._map_dietary_habits(train_df, train_df)
            test_df = self._map_dietary_habits(test_df, train_df)

            train_df = self._map_degree_values(train_df, train_df)
            test_df = self._map_degree_values(test_df, train_df)

            train_df = self._map_sleep_duration(train_df, train_df)
            test_df = self._map_sleep_duration(test_df, train_df)

             # 6. drop 'Name' if present (not used as feature)
            if 'Name' in train_df.columns:
                train_df = train_df.drop(columns=['Name'])
            if 'Name' in test_df.columns:
                test_df = test_df.drop(columns=['Name'])
            if 'id' in train_df.columns:
                train_df = train_df.drop(columns=['Name'])
            if 'id' in test_df.columns:
                test_df = test_df.drop(columns=['Name'])

            # split inputs and target
            if  TARGET_COLUMN not in train_df.columns:
                raise Exception(f"Target column '{TARGET_COLUMN}' not present in train data")
            X_train = train_df.drop(columns = [TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns = [TARGET_COLUMN]) if TARGET_COLUMN in test_df.columns else test_df.copy()
            y_test = test_df[TARGET_COLUMN].values if TARGET_COLUMN in test_df.columns else None

            ##### Tranformer and transforms
            logging.info('Starting data transformation')
            preprocessor = self.get_data_transformer_object(X_train)
            logging.info('Got the preprocessor object')

            logging.info('Initializing tranformation for training data')
            X_train_arr = preprocessor.fit_transform(X_train)
            logging.info('Initializing tranformation for testing data')
            X_test_arr = preprocessor.transform(X_test)
            logging.info('Transformation done to Train and test df')

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            if y_test is not None:
                test_arr = np.c_[X_test_arr, np.array(y_test)]
            else:
                test_arr = X_test_arr
            logging.info('feature target concatentation done for train-test df')

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info('Saving transformation object and transformed files.')

            logging.info('Data Transformation completed successfully')
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e


