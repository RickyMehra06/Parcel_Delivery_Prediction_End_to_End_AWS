from parcel.pipeline.training_pipeline import start_training_pipeline
from parcel.exception import ParcelDeliveryException
import os, sys

if __name__=="__main__":
    try:
        start_point = start_training_pipeline()
        print(start_point)

    except Exception as e:
       raise ParcelDeliveryException(e, sys)