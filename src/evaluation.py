import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class Evaluation(ABC):
    
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        
        pass
    
class MSE(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Caculatinng MSE Scores")
            mse=mean_squared_error(y_true, y_pred)
            logging.info("MSE:{}.format(mse)")
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE score:{}".format(e))
            raise e
        
class R2(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2=r2_score(y_true, y_pred)
            logging.info("R2 Score:{}.format(r2)")
            return r2
        except Exception as e:
            logging.error("Error in claculating r2 score {}.format(e)")
            raise e
            
class RMSE(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Caculatinng RMSE Scores")
            rmse=mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE:{}.format(rmse)")
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE score:{}".format(e))
            raise e