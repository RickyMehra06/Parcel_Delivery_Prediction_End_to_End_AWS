from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path:str
    train_file_path:str
    test_file_path:str
    
@dataclass
class DataValidationArtifact:
    data_validation_artifact:str

@dataclass
class DataTransformationArtifact:
    transformed_train_path:str
    transformed_test_path:str

    description_transformer_object_path:str
    target_encoder_object_path:str
    robust_scaler_object_path:str
    
@dataclass
class ModelTrainerArtifact:
    model_path:str
    f1_score_train:float 
    f1_score_test:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float

@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str    
    saved_model_dir:str 