from parcel.predictor import ModelResolver
from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel.entity import artifact_entity, config_entity
from parcel.utils import load_object, save_object
import os,sys

class ModelPusher:
    def __init__(self, model_pusher_config: config_entity.ModelPusherConfig,
                        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                        model_trainer_artifact : artifact_entity.ModelTrainerArtifact):
        
        try:
            logging.info(f"{'>>'*10} Stage 06- Model Pusher Initiated {'<<'*10}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(saved_models_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def initiate_model_pusher(self)-> artifact_entity.ModelPusherArtifact:
        try:
            #loading objects from transformation artifact
            logging.info(f"Loading model and transformer obejects from Stage_03")

            model = load_object(file_path=self.model_trainer_artifact.model_path)
            description_transformer = load_object(file_path=self.data_transformation_artifact.description_transformer_object_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_object_path)
            robust_scaler_object = load_object(file_path=self.data_transformation_artifact.robust_scaler_object_path)


            # Saving objects into Artifact's Pusher directory as currently trained model is better than previously trained model
            logging.info(f"Saving model and transformer objects into model_pusher directory")

            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.description_transformer_pusher_path, obj=description_transformer)
            save_object(file_path=self.model_pusher_config.target_encoder_pusher_path, obj=target_encoder)
            save_object(file_path=self.model_pusher_config.robust_scaler_pusher_path, obj=robust_scaler_object)


            # Saving model and transformer objects by creating latest in 'saved_model' folder
            logging.info(f"Saving model and transformer objects by creating latest folder into 'saved_model' folder")
            
            model_latest_path=self.model_resolver.get_latest_save_model_path()
            description_transformer_latest_path = self.model_resolver.get_latest_save_transformer_path(config_entity.DESCRIPTION_TRANSFORMER_OBJECT_FILE_NAME)
            target_encoder_latest_path = self.model_resolver.get_latest_save_transformer_path(config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
            robust_scaler_latest_path = self.model_resolver.get_latest_save_transformer_path(config_entity.ROBUST_SCALER_OBJECT_FILE_NAME)
            
            save_object(file_path=model_latest_path, obj=model)
            save_object(file_path=description_transformer_latest_path, obj=description_transformer)
            save_object(file_path=target_encoder_latest_path, obj=target_encoder)
            save_object(file_path=robust_scaler_latest_path, obj=robust_scaler_object)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.model_pusher_dir,
                                                        saved_model_dir=self.model_pusher_config.model_pusher_dir)

            logging.info(f"Model pusher artifact: {model_pusher_artifact}\n")
            return model_pusher_artifact

        except Exception as e:
            raise ParcelDeliveryException(e, sys)