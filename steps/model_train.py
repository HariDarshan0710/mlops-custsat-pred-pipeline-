import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client 
from src.model_dev import LinearRegressionModel
    # HyperParameterTunner,
    # LightGBMModel,
    
    # RandomForestModel,
    # XGBoostModel,



experiment_tracker=Client().active_stack.experiment_tracker




@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train:pd.DataFrame,
                X_test:pd.DataFrame,
                y_train:pd.Series,
                y_test:pd.Series,
                config:ModelNameConfig)->RegressorMixin:
    # logging.info("Training model")
    # pass
    try:
        model = None
        # tuner = None

        # if config.model_name == "lightgbm":
        #     mlflow.lightgbm.autolog()
        #     model = LightGBMModel()
        # elif config.model_name == "randomforest":
        #     mlflow.sklearn.autolog()
        #     model = RandomForestModel()
        # elif config.model_name == "xgboost":
        #     mlflow.xgboost.autolog()
        #     model = XGBoostModel()
        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train,)
            return trained_model
        else:
            raise ValueError("Model {} name not supported".format(config.model_name))

        # tuner = HyperParameterTunner(model, X_train, y_train, X_test, y_test)

        # if config.fine_tuning:
        #     best_params = tuner.optimize()
        #     trained_model = model.train(X_train, y_train, **best_params)
        # else:
        #     trained_model = model.train(X_train, y_train)
        # return trained_model
    except Exception as e:
        logging.error(e)
        raise e