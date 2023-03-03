# Parcel Delivery Prediction:

![](https://github.com/RickyMehra06/Parcel_Delivery_Prediction_End_to_End_AWS/blob/main/media/Parcel_Delivery_GIF.gif)

The repository consists of files required for end to end implementation and deployment of Parcel Delivery Prediction web application using machine learning created with Flask and deployed on the AWS platform for both Instance prediction and Batch file prediction.


## Problem Statement:

Amazon US facilitates Indian sellers to sell their products in the US marketplace through their seller-fulfilled channels using a program called Amazon Global Selling. An Amazon seller in India is selling his books at Amazon's US marketplace.

The default delivery date appears at the buyer's portal between 14-28 days since order is being fulfilled from outside US (India). The seller in India is using USPS as their deliverty service provider.

Usually, it takes at least 5 days to reach the parcel at the USPS center at New Jersey and then onwards 6th day, the buyer can see the tracking details of his parcel on the portal. Due to this blank spot of 5 days, the orders are being canceled from the buyers end as they are unable to track their parcel within initial 5 days of order placed. However maximum 10 days are taken to deliver the parcel in any part of the USA which is way sooner than the delivery threshold of 14-28 days of Amazon's US marketplace.

The Amazon seller in India is suffering high order-cancellation rate as orders are being canceled while they are in transit. This not only downgrades the account's health matrics but also increases the huge loss in terms of cost.

In order to reduce the order-cancellation rate, the seller needs a parcel delivery prediction model so that the expected delivery date can be shared with the buyer before initiating the dispatch process.

## Challenges and other objectives
* Extensive feature engineering is required as dataset has limited relevant features to develope the model.
* Web scrapping is used to extract weather condition.
* Google Maps API is used to find out the Latitude and Longitude coordinates.
* Google Maps API is used to find out the travel distance and time required to deliver the parcel.
* The same webpage can be used for both Instance prediction and Batch file prediction.
* The model will evaluate the currently trained model with the previously trained model. If the currently trained model is better then saving the model and transformer objects for the future prediction.

## Features Details:

### Features in Parcel dataset file:
* order-id: Unqiue ID for each order
* product-name: Name of the product sold
* purchase-date: Date and Time of the order recived in the form of Datetime stamp
* quantity-purchased: Units of product sold
* currency: Currency used for transaction, USD for US marketplace
* ship-service-level: "Standard" for all orders
* buyer-name: Buyer's name (Truncated)
* ship-address-1: Buyer's address-1 (Truncated)
* ship-address-2: Buyer's address-2 (Truncated)
* ship-city: Buyer's city name
* ship-state: Buyer's State name
* ship-state-id: Buyer's State ID in two alphadigits
* postal-code: Buyer's 5 digit US postal code
* ship-country: Buyer's country name (US for all oders)
* ship-phone-number: Buyer's contact number (Truncated)
* temp: Temprature of buyer's potal code
* unit: Unit of temprature in °C of buyer's potal code
* description: 8 unique strings about weather condition of buyer's potal code
* prepcipitation: float number between 0-1.0
* humidity: float number between 0.05-1.0
* wind: Speed in km/h
* weight: Weights of parcel in KG
* service: Delivery serivce provider name (USPS) for all orders
* tracking-number: Unqiue 27 digits alhanumeric ID for each order
* days-taken: Output feature (Number of days taken to deliver the parcel)


### Features in Coordinates dataset file:

* ship-city: Buyer's city name
* ship-state: Buyer's State name
* ship-state-id: Buyer's State ID in two alphadigits
* region: 5 unique US region name in string
* postal-code: Buyer's 5 digit US postal code
* latitude: Latitude coordinate of the buyer's postal code
* longitude: Longitude coordinate of the buyer's postal code
* distance: Travel distance between New Jersey's USPS center and the buyer's postal code in km
* duration: Time required to travel distance between New Jersey's USPS center and the buyer's postal code in hours



## Model architecture:

* RandomForest classifier is used to train the model to predict expected parcel deivery date.
* Choose radio button to select the Instance prediction or Batch file prediction as both can be performed using same HTML webpage.
* For instance prediction, user can provide required inputs on the HTML webpage and on click of the predict button to get the connection date and delivery date.
* For batch prediction, user can provide the input file path in .xlsx format and output file directory to get the predicted batch file.

![ML_Dev_Steps](https://github.com/RickyMehra06/Parcel_Delivery_Prediction_End_to_End_AWS/blob/main/media/Parcel_delivery_Architecture.jpg)


### Step 1 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 2 - To Run .py files for prediction

```bash
python main.py to train and evaluate the model

python app.py for both Instance prediction and Batch file prediction

```
