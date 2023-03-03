from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel import utils

from parcel.entity import config_entity
from parcel.entity import artifact_entity

from datetime import datetime
import os, sys
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*10} Stage 01- Data Ingestion initiated {'<<'*10}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ParcelDeliveryException(e, sys)
    
    def initiate_data_ingestion(self)-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Loading data from Mongodb")

            # Loading parcel_dataset from Mongodb
            data = utils.get_collection_as_dataframe(
                database_name= config_entity.DATABASE_NAME, 
                collection_name= config_entity.COLLECTION_NAME)
            
            data_ingestion_dir = os.path.join(self.data_ingestion_config.data_ingestion_dir)
            os.makedirs(data_ingestion_dir, exist_ok=True)

            data.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Loaded data saved into fetaure_store_file_path")


            # Loading coordinate details from Mongodb
            coordinates_df = utils.get_collection_as_dataframe(
                database_name= config_entity.DATABASE_NAME, 
                collection_name= config_entity.COORDINATES_COLLECTION_NAME)

            df = pd.merge(data, coordinates_df[["postal-code", "latitude", "longitude", "distance", "duration"]], 
                                            on = 'postal-code', how ='left')
            
            df.to_excel(self.data_ingestion_config.merged_file_path, index=False, header=True)
            logging.info(f"Merged Latitude and Longitude details and saved into merged_dataset")

            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42, stratify = df['days-taken'])

            logging.info(f"Saving data into train_file_path and test_file_path")
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            # prepare artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_path = self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path)
            
            logging.info(f"Stage 01- Data ingestion artifact: {data_ingestion_artifact}\n")
            return data_ingestion_artifact

        except Exception as e:
            raise ParcelDeliveryException(e, sys)


    



