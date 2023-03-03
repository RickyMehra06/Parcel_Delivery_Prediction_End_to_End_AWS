from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel.entity import config_entity, artifact_entity
from parcel import utils
import os, sys
import pandas as pd
import numpy as np
from typing import Optional
from typing import Dict
from parcel.entity.config_entity import TARGET_COLUMN

import warnings
warnings.filterwarnings('ignore')

#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class DataTransformation:
    def __init__(self, 
                    data_transformation_config: config_entity.DataTransformationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*10} Stage-03 Data Transformation Initiated {'<<'*10}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            
            self.drop_cols = ['order-id', 'product-name', 'currency', 'ship-service-level', 'buyer-name', 'ship-address-1', 
                                'ship-address-2', 'ship-city', 'ship-state', 'ship-state-id', 'ship-country', 
                                'ship-phone-number', 'unit', 'service', 'tracking-number','postal-code','latitude', 'longitude',
                                 'purchase_date', 'dispatched_date', 'connection_date']

            self.reset_cols = ['quantity-purchased', 'weight_transformed', 'dispatched_days', 'connection_days', 'description',
                                'temp', 'prepcipitation', 'humidity', 'wind', 'distance', 'duration','days-taken']

            self.robustscaler = RobustScaler()
    
        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def drop_missing_values(self, df)->pd.DataFrame:
        try:
            col_list = df.columns[:-4]
            initial_rows = df.shape[0]
            missing_values_count = df.iloc[:,:-4].isnull().values.sum()

            if missing_values_count > 0:
                df.dropna(how='any', subset= col_list, axis=0, inplace=True)
                df.reset_index(drop=True)

                final_rows = df.shape[0]
                dropped_rows = initial_rows - final_rows
                logging.info(f"There were {missing_values_count} missing values in {dropped_rows} rows and are removed from the dataset.")
                return df

            logging.info(f"There is no missing value in the dataset.")   
            return df

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def drop_duplicate_rows(self,df)->pd.DataFrame:
        try:
            duplicate_rows_count = len(df[df.duplicated()])

            if duplicate_rows_count > 0:
                df.drop_duplicates(keep ='first', inplace=True)
                df.reset_index(drop=True)
                logging.info(f"{duplicate_rows_count} number of duplicate rows are removed from the dataset.")
                return df

            logging.info(f"There is no duplicate rows in the dataset.")   
            return df

        except Exception as e:
            raise ParcelDeliveryException(e, sys)


    def feature_encoding(self, df, column_name)->Dict:
        try:
            temp_list = list(sorted(df[column_name].unique()))
            logging.info(f"{column_name} has dictionary: {dict(zip(temp_list, range(len(temp_list))))}")
            return dict(zip(temp_list, range(len(temp_list))))               

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def drop_columns(self, df, drop_columns_list)->pd.DataFrame:
        try:
            df.drop(drop_columns_list, axis =1, inplace =True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Features not required for the model have been removed from the dataframe.")   
            return df

        except Exception as e:
                raise ParcelDeliveryException(e, sys)


    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:

        try:
            logging.info("-----Transforming Train dataset-----")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            
            train_df = self.drop_missing_values(df=train_df)
            train_df = self.drop_duplicate_rows(df=train_df)
            
            train_df = utils.split_date_feature(df=train_df, column_name='purchase-date')
            train_df = utils.handling_time(df=train_df, column_name='purchase_date')
            train_df = utils.connection_days(df=train_df, column_name='dispatched_date')  
            train_df = utils.weight_transformation(df=train_df, column_name='weight')          
            train_df = utils.get_lat_long(df=train_df, column_name= 'postal-code')
            train_df = utils.distance_matrix(df=train_df, lat_col= 'latitude', long_col='longitude')
            
            train_df['wind'] = train_df['wind'].str.split('k').str[0]            

            description_encoder = self.feature_encoding(df=train_df, column_name='description')
            train_df['description'] = train_df['description'].map(description_encoder)

            target_encoder = self.feature_encoding(df=train_df, column_name='days-taken')
            train_df['days-taken'] = train_df['days-taken'].map(target_encoder)
        
            train_df.drop(self.drop_cols, axis =1, inplace = True)      
            train_df = train_df.reindex(self.reset_cols, axis =1)
            train_df.iloc[:,5:-1] = self.robustscaler.fit_transform(train_df.iloc[:,5:-1])


            logging.info("-----Transforming Test dataset-----")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path) 

            test_df = self.drop_missing_values(df=test_df)
            test_df = self.drop_duplicate_rows(df=test_df)
            
            test_df = utils.split_date_feature(df=test_df, column_name='purchase-date')
            test_df = utils.handling_time(df=test_df, column_name='purchase_date')
            test_df = utils.connection_days(df=test_df, column_name='dispatched_date')
            test_df = utils.weight_transformation(df=test_df, column_name='weight')
            test_df = utils.get_lat_long(df=test_df, column_name= 'postal-code')
            test_df = utils.distance_matrix(df=test_df, lat_col= 'latitude', long_col='longitude')

            test_df['wind'] = test_df['wind'].str.split('k').str[0]            

            test_df['description'] = test_df['description'].map(description_encoder)
            test_df['days-taken'] = test_df['days-taken'].map(target_encoder)

            test_df.drop(self.drop_cols, axis =1, inplace = True)    
            test_df = test_df.reindex(self.reset_cols, axis =1) 
            test_df.iloc[:,5:-1] = self.robustscaler.transform(test_df.iloc[:,5:-1])

            
            utils.save_data(file_path= self.data_transformation_config.transformed_train_path , df= train_df)
            utils.save_data(file_path= self.data_transformation_config.transformed_test_path , df= test_df)

            utils.save_object(file_path=self.data_transformation_config.description_transformer_object_path, obj=description_encoder)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_object_path, obj=target_encoder)
            utils.save_object(file_path=self.data_transformation_config.robust_scaler_object_path, obj=self.robustscaler)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(                    
                    transformed_train_path = self.data_transformation_config.transformed_train_path,
                    transformed_test_path = self.data_transformation_config.transformed_test_path,
                    description_transformer_object_path = self.data_transformation_config.description_transformer_object_path,
                    target_encoder_object_path = self.data_transformation_config.target_encoder_object_path,
                    robust_scaler_object_path = self.data_transformation_config.robust_scaler_object_path
                    )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}\n")
            return data_transformation_artifact

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

        