from sklearn.metrics import f1_score
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.logger import logging
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object, load_numpy_array_data
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.entity.s3_estimator import ProjEstimator

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
    
    def get_best_model(self) -> Optional[ProjEstimator]:
        """
        Description: This method is used to get model from production stage
        Output: Returns model object if available in s3 storage
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj_estimator = ProjEstimator(bucket_name=bucket_name,
                                           model_path=model_path)
            
            if proj_estimator.is_model_present(model_path=model_path):
                return proj_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name: evaluate model
        Description: This function is used to evaluate trained model with production model and choose the best model
        Output: Return bool value based on validation results
        """
        try:
            logging.info("Starting model evaluation process...")

            ## load transformed test data (.npy)
            test_data = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loaded transformed test array from : {self.data_transformation_artifact.transformed_test_file_path}")

            # split features and target
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # load trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info('Loaded trained model successfully')
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1 Score for this model: {trained_model_f1_score}")

            ## Evaluate trained model
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info(f"Production model found - comparing performance..")
                y_pred_best = best_model.predict(X_test)
                best_model_f1_score = f1_score(y_test, y_pred_best)
                logging.info(f"F1-score Production model: {best_model_f1_score} | F1-Score-New Trained model: {trained_model_f1_score}")

            # Compute comparison and acceptance
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_model_score
            difference = trained_model_f1_score - tmp_best_model_score

            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=is_model_accepted,
                                           difference=difference)
            
            logging.info(f"Model Evaluation Result: {result}")
            return result
        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function is used to initiate all steps of the model evaluation
        Output: Returns model evaluation artifact
        """
        try:
            print('----------------------------------------------------------------------------------------')
            logging.info("Initialized Model evaluation component")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
