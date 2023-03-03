from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel.config import mongo_client
from parcel.config import gmaps

import pandas as pd
import numpy as np
import os, sys
import yaml
import pickle
from datetime import datetime

import googlemaps 

#google_api_key = os.getenv("GOOGLE_API_KEY")
#gmaps = googlemaps.Client(key=google_api_key) 

IND_holiday_list = ['2022-10-02', '2022-10-05','2022-10-24','2022-11-08','2022-12-25']
US_holiday_list = ['2022-10-10', '2022-11-24', '2022-12-25', '2023-01-01']


def get_collection_as_dataframe(database_name:str, collection_name:str)-> pd.DataFrame:
    try:
        logging.info(f"Reading data from Mongodb from database-{database_name} and collection-{collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        
        if "_id" in df.columns:
            logging.info("Removing _id column from the dataframe")
            df.drop("_id", axis=1, inplace =True)
        logging.info(f"Rows and colums: {df.shape}")
        return df

    except Exception as e:
        raise ParcelDeliveryException(e, sys)
    
def write_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_obj:
            yaml.dump(data, file_obj)

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def save_object(file_path:str, obj:object)->None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"{file_path.split('/')[-1]} object has been saved thorough utils")
    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def load_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
    
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def save_data(file_path:str, df)-> None:
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        df.to_csv(file_path, index= False, header =True)

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def load_data(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The dataframe {file_path} does not exist.")

        return pd.read_csv(file_path)

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def split_date_feature(df, column_name):
        try:
            df['purchase_date'] = df[column_name].str.split('T').str[0]
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])   # converting datetime to add business days further    
            
            df.drop(column_name, axis =1, inplace=True)
            df.reset_index(drop=True)
            logging.info(f"{column_name} is converted into date column.")
            return df

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

def handling_time(df, column_name):
    try:    
        business_days = pd.tseries.offsets.CustomBusinessDay(n=2, weekmask='Mon Tue Wed Thu Fri Sat', holidays=IND_holiday_list)
        df['dispatched_date'] = df[column_name] + business_days
        df['dispatched_days'] = (df['dispatched_date'] - df[column_name]).dt.days
        logging.info(f"Dispatched dates are generated.")
        return df

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def connection_days(df, column_name):   
    try: 
        business_days = pd.tseries.offsets.CustomBusinessDay(n=3, weekmask='Mon Tue Wed Thu Fri Sat', holidays=US_holiday_list)
        df['connection_date'] = df[column_name] + business_days
        df['connection_days'] = (df['connection_date'] - df[column_name]).dt.days
        logging.info(f"Connection dates are generated.")
        return df

    except Exception as e:
        raise ParcelDeliveryException(e, sys)

def weight_transformation(df, column_name):
        try:
            df['weight_transformed'] =  np.where(df[column_name]>2.5,6, 
                                        np.where(df[column_name]>2.0,5, 
                                        np.where(df[column_name]>1.5,4, 
                                        np.where(df[column_name]>1.0,3,
                                        np.where(df[column_name]>0.5,2,1)))))
            df.drop(column_name, axis =1, inplace=True)
            logging.info(f"Weight feature has been bucketized into 500 grams each.")
            return df

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

def get_lat_long(df, column_name):
    
    temp_df = df[df['latitude'].isna()]    
    if temp_df.empty:
        return df
    
    else:        
        #gmaps = googlemaps.Client(key=api_key)
        lat_list = []
        long_list = []

        for postal_code in temp_df['postal-code']:
            # Geocode the postal code using the Google Maps API
            geocode_result = gmaps.geocode(postal_code)

            if geocode_result:
                # Extract the latitude and longitude from the geocode result
                lat = geocode_result[0]['geometry']['location']['lat']
                long = geocode_result[0]['geometry']['location']['lng']

                lat_list.append(lat)
                long_list.append(long)

            else:
                print(f'Could not find coordinates for this postal code {postal_code}')
                lat_list.append(0)
                long_list.append(0)        

        temp_df['latitude'] = lat_list
        temp_df['longitude'] = long_list
        
        df = df.combine_first(temp_df)
        df.drop(df[df['latitude']==0].index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Latitude and Longitude coordinates are featched through Google Map API.")
        return df

def distance_matrix(df, lat_col, long_col):
    
    #Partner USPS center is at New Jersey where the items are received, scanned and then sent out for the delivery in variours parts of the USA.
    origin_lat = str(40.5784)        # New Jersey' partnered USPS center latitude coordinate
    origin_long = str(-74.2833)      # New Jersey' partnered USPS center longitude coordinate
    
    temp_df = df[df['distance'].isna()]    
    if temp_df.empty:
        return df
    
    else:       
        #gmaps = googlemaps.Client(key=api_key)
        distance_list = []
        duration_list = []

        for i,j in zip(temp_df[lat_col], temp_df[long_col]):

            temp_result = gmaps.distance_matrix([origin_lat+" "+origin_long],[str(i)+" "+str(j)], 
                                                mode='driving')['rows'][0]['elements'][0]

            if temp_result['status']=='OK':          
                distance_list.append(temp_result['distance']['text'])
                duration_list.append((temp_result['duration']['value'])/3600)   # Converted into hours

            else:
                print(f'Could not find Distance for this latitude {i} & longitude {j}.')
                distance_list.append(0)
                duration_list.append(0)    
        
        temp_df['distance'] = distance_list
        temp_df['duration'] = duration_list
                
        df = df.combine_first(temp_df) 
        df['distance'] = df['distance'].apply(lambda x: x.split(" ")[0].replace(",","") if isinstance(x, str) else x)
        df.drop(df[df['duration']==0].index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Travel distance and time duration are generated through Google Map API.")
        return df

def get_lat_long_instance(postal_code):    
      
    #gmaps = googlemaps.Client(key=google_api_key)
    geocode_result = gmaps.geocode(postal_code)

    if geocode_result:
        # Extract the latitude and longitude from the geocode result
        lat = geocode_result[0]['geometry']['location']['lat']
        long = geocode_result[0]['geometry']['location']['lng']            
    else:
        logging.info(f'Could not find lat-long coordinates for the postal code- {postal_code}')
        lat = 0
        long= 0

    return lat, long

def distance_matrix_instance(destination_lat, destination_long):
    
    #Partner USPS center is at New Jersey where the items are received, scanned and then sent out for the delivery in variours parts of the USA.
    origin_lat = str(40.5784)        # New Jersey' partnered USPS center latitude coordinate
    origin_long = str(-74.2833)      # New Jersey' partnered USPS center longitude coordinate

    #gmaps = googlemaps.Client(key=google_api_key)
    
    temp_result = gmaps.distance_matrix([origin_lat+" "+origin_long], [str(destination_lat)+" "+str(destination_long)], 
                                        mode='driving')['rows'][0]['elements'][0]

    if temp_result['status']=='OK':          
        distance = temp_result['distance']['text']
        distance = distance.split(" ")[0]
        distance = distance.replace(",","")

        duration = (temp_result['duration']['value'])/3600   # Converted into hours            
    else:
        print(f'Could not find Distance for the latitude {destination_lat} & longitude {destination_long}.')
        distance = 0
        duration = 0
    
    return distance, duration