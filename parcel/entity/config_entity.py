from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
import os, sys
from datetime import datetime

DATABASE_NAME = "parcel_delivery_db"
COLLECTION_NAME ="parcel_data_collection"
COORDINATES_COLLECTION_NAME = "coordinates_data_collection"

FILE_NAME = "original_dataset.csv"
MERGED_FILE_NAME = "merged_dataset.xlsx"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "days-taken"

MODEL_FILE_NAME = "model.pkl"

DESCRIPTION_TRANSFORMER_OBJECT_FILE_NAME = "description_transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME= "target_encoder.pkl"
ROBUST_SCALER_OBJECT_FILE_NAME = "robust_scaler_object.pkl"

class TrainingPipelineConfig:
    
    def __init__(self):
        try: 
            self.artifact_dir = os.path.join(os.getcwd(), "Artifact", f"{datetime.now().strftime('%Y-%m-%d__%H:%M:%S')}")

        except Exception as e:
            raise ParcelDeliveryException(e,sys) 

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")

            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, FILE_NAME)
            self.merged_file_path = os.path.join(self.data_ingestion_dir, MERGED_FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, TEST_FILE_NAME)    

            self.test_size = 0.3

        except Exception  as e:
            raise ParcelDeliveryException(e,sys) 


class DataValidationConfig:
     def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, "validation_report.yaml")

            self.missing_threshold:float = 0.2
            self.base_file_path = "/config/workspace/datasets/Parcel_dataset_1.xlsx"     

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")

        self.transformed_train_path = os.path.join(self.data_transformation_dir, "Transformed", TRAIN_FILE_NAME)
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "Transformed", TEST_FILE_NAME)

        self.description_transformer_object_path = os.path.join(self.data_transformation_dir, "Transformers", DESCRIPTION_TRANSFORMER_OBJECT_FILE_NAME)
        self.target_encoder_object_path = os.path.join(self.data_transformation_dir, "Transformers", TARGET_ENCODER_OBJECT_FILE_NAME)        
        self.robust_scaler_object_path = os.path.join(self.data_transformation_dir, "Transformers", ROBUST_SCALER_OBJECT_FILE_NAME)

class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")

        self.model_path = os.path.join(self.model_trainer_dir,"model", MODEL_FILE_NAME)
        self.expected_error = 0.6
        self.overfitting_threshold = 0.1

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.1

class ModelPusherConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        self.saved_model_dir = os.path.join("saved_models") 
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)

        self.description_transformer_pusher_path = os.path.join(self.model_pusher_dir,"Transformers", DESCRIPTION_TRANSFORMER_OBJECT_FILE_NAME)
        self.target_encoder_pusher_path = os.path.join(self.model_pusher_dir, "Transformers", TARGET_ENCODER_OBJECT_FILE_NAME)
        self.robust_scaler_pusher_path = os.path.join(self.model_pusher_dir,"Transformers", ROBUST_SCALER_OBJECT_FILE_NAME)
 