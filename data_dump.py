import pymongo
import pandas as pd
import json

from parcel.config import mongo_client

DATA_FILE_NAME = "/config/workspace/datasets/Parcel_dataset_1.xlsx"
COORDINATES_FILE_NAME = "/config/workspace/datasets/Coordinates_dataset.xlsx"

DATABASE_NAME = "parcel_delivery_db"
COLLECTION_NAME ="parcel_data_collection"
COORDINATES_COLLECTION_NAME = "coordinates_data_collection"

if __name__=="__main__":
    df = pd.read_excel(DATA_FILE_NAME)
    df.reset_index(drop=True,inplace=True)
    print(f"The size of Parcel dataset is: {df.shape}")

    #Convert dataframe to json to dump these record in mongodb
    json_record = list(json.loads(df.T.to_json()).values())

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print(f"{df.shape[0]} records have been uploaded into Mongodb collection:-> {COLLECTION_NAME}")

    data = pd.read_excel(COORDINATES_FILE_NAME)
    data.reset_index(drop=True,inplace=True)
    print(f"The size of Coordinates dataset is: {data.shape}")

    #Convert dataframe to json to dump these record in mongodb
    json_record = list(json.loads(data.T.to_json()).values())

    mongo_client[DATABASE_NAME][COORDINATES_COLLECTION_NAME].insert_many(json_record)
    print(f"{data.shape[0]} records have been uploaded into Mongodb collection:-> {COORDINATES_COLLECTION_NAME}")
