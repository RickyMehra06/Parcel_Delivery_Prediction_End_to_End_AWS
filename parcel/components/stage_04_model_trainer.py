from parcel.exception import ParcelDeliveryException
from parcel.logger import logging
from parcel.entity import artifact_entity, config_entity
from parcel import utils
import os, sys
from typing import Dict

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class ModelTrainer:
    def __init__(self,
                    model_trainer_config: config_entity.ModelTrainerConfig,
                    data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*10} Stage-04 Model Trainer Initiated {'<<'*10}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def fine_tune(self,x,y)->Dict:
        try:
            param_grid = {'criterion' : ['gini', 'entropy'],
                        'n_estimators' : [10],
                        'max_depth' : [3],
                        'max_features':  [3],
                        'min_samples_leaf': [3],
                        'oob_score' : [False],
                        'random_state': [42]
                        }
            #best_params_:{'criterion': 'gini', 'max_depth': 7, 'max_features': 5, 'min_samples_leaf': 5, 'n_estimators': 100, 'oob_score': False, 'random_state': 42}
            logging.info(f">>>>>>>>>>  Tunning initiated  <<<<<<<<<<<")

            grid = GridSearchCV(RandomForestClassifier(), param_grid,  scoring='accuracy')
            grid.fit(x,y)
            logging.info(f"best_params_ are:{grid.best_params_}")
            return grid.best_params_
            
        except Exception as e:
            raise ParcelDeliveryException(e, sys) 
    
    def train_model(self,x,y,best_params):
        try:
            clf = RandomForestClassifier(**best_params)
            clf.fit(x,y)
            return clf

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

    def initiate_model_trainer(self)-> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test data for model trainer.")
            train_data = utils.load_data(self.data_transformation_artifact.transformed_train_path)
            test_data = utils.load_data(self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test data.")
            x_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]
            x_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]

            try:
                logging.info(f"Fine tuning the model using GridSearchCV")
                best_params = self.fine_tune(x=x_train, y=y_train)

            except Exception as e:
                raise ParcelDeliveryException(e, sys)

            logging.info(f"Training the model")
            model = self.train_model(x_train, y_train, best_params)

            logging.info(f"Calculating f1_score for train dataset")
            y_pred_train = model.predict(x_train)
            f1_score_train = f1_score(y_train, y_pred_train, average="weighted")
            
            logging.info(f"Calculating f1_score for test dataset")
            y_pred_test = model.predict(x_test)
            f1_score_test = f1_score(y_test, y_pred_test, average="weighted")

            logging.info(f"train f1 score:{f1_score_train} and test f1 score: {f1_score_test}")
            
            #check for overfitting or underfiiting or expected error

            logging.info(f"Checking if our model is underfitting or not")
            if f1_score_test < self.model_trainer_config.expected_error:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_error} however model actual score is: {f1_score_test}")
            
            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_score_train - f1_score_test)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test accuracy diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
                
            #save the trained model
            logging.info(f"Saving the model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare model trainer artifact
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                f1_score_train = f1_score_train, 
                f1_score_test = f1_score_test)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}\n")

            return model_trainer_artifact

        except Exception as e:
            raise ParcelDeliveryException(e, sys)

            
