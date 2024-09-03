import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStratergy, DataPreProcessStratergy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[ Annotated[pd.DataFrame, "X_train"],
                                         Annotated[pd.DataFrame, "X_test"],
                                         Annotated[pd.Series, "y_train"],
                                         Annotated[pd.Series, "y_test"],]:
    try:
        process_strategy=DataPreProcessStratergy()
        data_cleaning=DataCleaning(df, process_strategy)
        processed_data=data_cleaning.handle_data()
        
        divide_strategy=DataDivideStratergy()
        data_cleaning=DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test=data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e 