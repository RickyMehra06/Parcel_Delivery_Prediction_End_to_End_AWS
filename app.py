from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel.entity import config_entity
from parcel import utils
from parcel.predictor import ModelResolver

import numpy as np
import pandas as pd
import os, sys
from datetime import datetime, timedelta
import pickle
from flask import Flask, request, jsonify, url_for, render_template
from requests_html import HTMLSession

import warnings
warnings.filterwarnings('ignore')

session= HTMLSession()

IND_holiday_list = ['2022-10-02', '2022-10-05','2022-10-24','2022-11-08','2022-12-25']
US_holiday_list  = ['2022-10-10', '2022-11-24', '2022-12-25', '2023-01-01']

drop_cols = ['product-name','currency','ship-service-level','buyer-name','ship-address-1','ship-address-2',
            'ship-city','ship-state','ship-state-id','ship-country','ship-phone-number','unit', 'service', 'tracking-number',
             'postal-code','latitude','longitude','dispatched_date']

reset_cols = ['order-id','purchase_date','connection_date',
                'quantity-purchased','weight_transformed','dispatched_days','connection_days','description',
                'temp', 'prepcipitation', 'humidity', 'wind', 'distance', 'duration']


app = Flask(__name__)

logging.info(f"Loading Latest model from saved_models folder!!")
latest_dir = ModelResolver().get_latest_dir_path()

model_path= os.path.join(latest_dir,"model",config_entity.MODEL_FILE_NAME)
parcel_model = utils.load_object(file_path=model_path)


logging.info(f"Loading Latest trasnformers from saved_models folder!!")

description_path = os.path.join(latest_dir,"transformers",config_entity.DESCRIPTION_TRANSFORMER_OBJECT_FILE_NAME)
target_encoder_path = os.path.join(latest_dir,"transformers",config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
robust_scaler_path = os.path.join(latest_dir,"transformers",config_entity.ROBUST_SCALER_OBJECT_FILE_NAME)

description_transformer = utils.load_object(description_path)
target_encoder = utils.load_object(target_encoder_path)
robust_scaler = utils.load_object(robust_scaler_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = [x for x in request.form.values()]
        print(data)
    except Exception as e:
        raise ParcelDeliveryException(e, sys)

    if len(data)>2:   # As instance prediction will have 4 user inputs
        try:
            logging.info(f"--------Instance Prediction initaited--------")
            logging.info(f"User inputs are: {data}")

            order_date_str = data[0]
            order_date = datetime.strptime(order_date_str, '%Y-%m-%d').date()

            filtered_data = []
            filtered_data.append(data[1])  # Quantity purcahsed
            filtered_data.append(data[2])  # Parcel weight               

            postal_code = data[3]

            dispatched_business_days = pd.tseries.offsets.CustomBusinessDay(n=2, weekmask='Mon Tue Wed Thu Fri Sat', holidays=IND_holiday_list)
            dispatched_date = (order_date + dispatched_business_days).date()
            dispatched_days = (dispatched_date - order_date).days

            filtered_data.append(dispatched_days)

            connection_business_days = pd.tseries.offsets.CustomBusinessDay(n=3, weekmask='Mon Tue Wed Thu Fri Sat', holidays=US_holiday_list)
            connection_date = (dispatched_date + connection_business_days).date()
            connection_days = (connection_date - order_date).days

            filtered_data.append(connection_days)

            session= HTMLSession()

            url = f'https://www.google.com/search?q=weather+in+{postal_code}+US'

            req = session.get(url)

            temp = req.html.find("span#wob_tm", first = True).text
            precipitation = req.html.find("div.wtsRwe", first= True).find("span#wob_pp", first= True).text
            precipitation = precipitation.replace("%","")
            humidity = req.html.find("div.wtsRwe", first= True).find("span#wob_hm", first= True).text
            humidity = humidity.replace("%","")
            wind = req.html.find("div.wtsRwe", first= True).find("span#wob_ws", first= True).text
            wind = wind.split(" ")[0]
            desc = req.html.find("div.VQF4g", first= True).find("span#wob_dc", first= True).text

            if 'sleet' in desc.lower():
                var = "Sleet"
            elif 'snow' in desc.lower():
                var = "Snow"
            elif 'fog' in desc.lower():
                var = "Fog"
            elif 'smoke' in desc.lower():
                var = "Smoke"
            elif 'rain' in desc.lower():
                var = "Rain"
            elif 'cloud' in desc.lower():
                var = "Cloudy"
            elif 'drizzle' in desc.lower():
                var = "Drizzle"
            elif 'haze' in desc.lower():
                var = "Haze"
            elif 'wind' in desc.lower():
                var = "Windy"
            elif 'sunny' in desc.lower():
                var = "Sunny"
            else:
                var = "Clear"
            desc =var

            filtered_data.append(desc)
            filtered_data.append(temp)
            filtered_data.append(precipitation)
            filtered_data.append(humidity)
            filtered_data.append(wind)

            filtered_data[4] = int((pd.Series(filtered_data[4]).map(description_transformer)).values)

            lat, long = utils.get_lat_long_instance(postal_code=postal_code)
            logging.info(f"Latitude: {lat} and Longitude: {long} for the postal code: {postal_code}.") 

            distance, duration = utils.distance_matrix_instance(destination_lat=lat, destination_long=long)
            logging.info(f"Distance: {distance} KM and Travel Time: {duration} Hours for the postal code: {postal_code}.") 

            if distance !=0:

                filtered_data.append(distance)
                filtered_data.append(duration)
                filtered_data = [float(x) for x in filtered_data] 

                final_input= np.array(filtered_data[:5]).reshape(1,-1)
                scaled_input = robust_scaler.transform(np.array(filtered_data[5:]).reshape(1,-1))
                final_input = np.concatenate((final_input, scaled_input), axis=1)

                output = parcel_model.predict(final_input)[0]
                invert_target_encoder = dict(map(reversed, target_encoder.items()))
                predicted_output = int((pd.Series(output).map(invert_target_encoder)).values)

                expected_delivery_date = order_date + timedelta(predicted_output)
                                
                output_text = (f"Instance Prediction: Parcel can be tracked from {connection_date} and \n"
                                f"Expected delivery date is {expected_delivery_date}")

                print("Instance Prediction")
                logging.info(f"Instance Prediction: Connection date: {connection_date} and Expected Delivery date: {expected_delivery_date}")
                return render_template('home.html', output_text=output_text)
            
            else:
                logging.info(f"Instance Prediction: Unable to fectch Travel distance or duration for the given postal.")
                return render_template('home.html', output_text="Instance Prediction: Please try again for other US postal code.")

        except Exception as e:
            raise ParcelDeliveryException(e, sys)
            logging.info(f"Please try with some other US postal code")
                
    else:
        try:
            logging.info(f"--------Batch Prediction initaited--------")
            logging.info(f"User input file name is: {data[0]}")
            
            INPUT_FILE_PATH = data[0]
            OUTPUT_DIR = data[1]

            temp_df = pd.read_excel(INPUT_FILE_PATH)

            # Loading coordinate details from Mongodb
            coordinates_df = utils.get_collection_as_dataframe(
                database_name= config_entity.DATABASE_NAME, 
                collection_name= config_entity.COORDINATES_COLLECTION_NAME)

            df = pd.merge(temp_df, coordinates_df[["postal-code", "latitude", "longitude", "distance", "duration"]], 
                                            on = 'postal-code', how ='left')

            df = utils.split_date_feature(df=df, column_name='purchase-date')
            df = utils.handling_time(df=df, column_name='purchase_date')
            df = utils.connection_days(df=df, column_name='dispatched_date')
            df = utils.weight_transformation(df=df, column_name='weight')
            df = utils.get_lat_long(df=df, column_name= 'postal-code')
            df = utils.distance_matrix(df=df, lat_col= 'latitude', long_col='longitude')
            df['wind'] = df['wind'].str.split('k').str[0]

            df['description'] = df['description'].map(description_transformer)
   
            df.drop(drop_cols, axis =1, inplace = True)      
            df = df.reindex(reset_cols, axis =1)
            df.iloc[:,8:-1] = robust_scaler.fit_transform(df.iloc[:,8:-1])

            # features: 'order-id','purchase_date','connection_date' needs to be skipped for model prediction
            prediction = parcel_model.predict(df.iloc[:,3:])   
            df["Prediction"] = prediction
            
            invert_target_encoder = dict(map(reversed, target_encoder.items()))
            df['Prediction'] = df['Prediction'].map(invert_target_encoder)

            df['Delivery_date'] = df['purchase_date'] + pd.to_timedelta(df['Prediction'], unit='d')
            df['Delivery_date'] = df['Delivery_date'].dt.date
            df['connection_date'] = df['connection_date'].dt.date

            temp_df = pd.merge(temp_df, df[['order-id','connection_date', 'Delivery_date', 'Prediction']], 
                                            on = 'order-id', how ='left')

            #temp_df.to_excel("/config/workspace/Output_folder/Output_file.xlsx", index=False, header=True)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logging.info(f"Output directory is given as : {OUTPUT_DIR}.")

            BATCH_PREDICTION_FILE_NAME = f"batch_prediction_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}.xlsx"
            batch_prediction_output_path = os.path.join(OUTPUT_DIR, BATCH_PREDICTION_FILE_NAME)

            logging.info(f"Output file path is : {batch_prediction_output_path}.")
            temp_df.to_excel(batch_prediction_output_path, index=False, header=True)
            
            logging.info(f"Batch Prediction file is saved at: {batch_prediction_output_path}.")
            return render_template('home.html', output_text=f"Batch Prediction file is saved at: {batch_prediction_output_path}")

        except Exception as e:
            raise ParcelDeliveryException(e, sys)
            logging.info(f"Review the batch prediction file.")
          
if __name__=="__main__":
    try:
        #app.run(debug=True)
        #app.run(host="0.0.0.0")
        app.run(host="0.0.0.0", port=8000)

    except Exception as e:
       raise ParcelDeliveryException(e, sys)