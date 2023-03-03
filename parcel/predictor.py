from parcel.exception import ParcelDeliveryException
from parcel.entity import artifact_entity, config_entity
import os, sys
from typing import Optional

class ModelResolver:
    def __init__(self, saved_models_registry  = "saved_models",
                        model_dir_name = "model",
                        transformer_dir_name = "transformers",
                        target_encoder_dir_name = "target_encoder"):
       
       self.saved_models_registry = saved_models_registry
       os.makedirs(saved_models_registry, exist_ok=True)

       self.model_dir_name = model_dir_name
       self.transformer_dir_name = transformer_dir_name
       self.target_encoder_dir_name = target_encoder_dir_name
        
    def get_latest_dir_path(self) -> Optional[str]:
        try:
            dir_names = os.listdir(self.saved_models_registry)
            if len(dir_names)==0:
                return None
            dir_names =  list(map(int,dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.saved_models_registry, f"{latest_dir_name}")
        
        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Pre-trained model is not available yet.")
            return os.path.join(latest_dir, self.model_dir_name, config_entity.MODEL_FILE_NAME)

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def get_latest_transformer_path(self, transformer_name):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Pre-transformer objects are not available yet.")
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_name)

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    # To get objects from the saved folder

    def get_latest_save_dir_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.saved_models_registry,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.saved_models_registry, f"{latest_dir_num+1}")

        except Exception as e:
            raise ParcelDeliveryException(e, sys)
    
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def get_latest_save_transformer_path(self, transformer_name):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, transformer_name)
        except Exception as e:
            raise ParcelDeliveryException(e, sys)


