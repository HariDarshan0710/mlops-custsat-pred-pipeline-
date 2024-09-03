from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df=ingest_df(data_path)
    X_train, X_test, y_train, y_test=clean_df(df)
    model=train_model( X_train, X_test, y_train, y_test)
    r2_score, rmse=evaluate_model(model, X_test, y_test)
    
# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
# from zenml.pipelines import pipeline

# docker_settings = DockerSettings(required_integrations=[MLFLOW])


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
# def train_pipeline(ingest_df, clean_df, model_train, evaluation):
#     """
#     Args:
#         ingest_data: DataClass
#         clean_data: DataClass
#         model_train: DataClass
#         evaluation: DataClass
#     Returns:
#         mse: float
#         rmse: float
#     """
#     df = ingest_df()
#     x_train, x_test, y_train, y_test = clean_df(df)
#     model = model_train(x_train, x_test, y_train, y_test)
#     mse, rmse = evaluation(model, x_test, y_test)