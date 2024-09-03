# import json

# # from .utils import get_data_for_test
# import os

# import numpy as np
# import pandas as pd
# # from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
# from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.steps import BaseParameters, Output

from .utils import get_data_for_test

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
# import pandas as pd

# # import os


# # from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
# #     MLFlowModelDeployer,
# # )
# # from zenml.integrations.mlflow.services import MLFlowDeploymentService
# # from zenml.pipelines import pipeline
# # from zenml.steps import BaseParameters, Output, step


# # requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

# # docker_settings= DockerSettings(required_integrations=[MLFLOW])
# requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
# @step(enable_cache=False)
# def dynamic_importer()->str:
#     data= get_data_for_test()
#     return data


# class DeploymentTriggerConfig(BaseParameters):
#     min_accuracy: float=0.5
    
# @step()
# def deployment_trigger(
#     accuracy: float,
#     config: DeploymentTriggerConfig,
# ):
#     return accuracy >=config.min_accuracy

# class MLFlowDeploymentLoaderStepParameters(BaseParameters):
#     """MLflow deployment getter parameters

#     Attributes:
#         pipeline_name: name of the pipeline that deployed the MLflow prediction
#             server
#         step_name: the name of the step that deployed the MLflow prediction
#             server
#         running: when this flag is set, the step only returns a running service
#         model_name: the name of the model that is deployed
#     """

#     pipeline_name: str
#     step_name: str
#     running: bool = True

# @step(enable_cache=False)
# def prediction_service_loader(
#     pipeline_name: str,
#     pipeline_step_name: str,
#     running: bool=True,
#     model_name: str="model",
# )->MLFlowDeploymentService:
#     """Get the prediction service started by the deployment pipeline.

#     Args:
#         pipeline_name: name of the pipeline that deployed the MLflow prediction
#             server
#         step_name: the name of the step that deployed the MLflow prediction
#             server
#         running: when this flag is set, the step only returns a running service
#         model_name: the name of the model that is deployed
#     """
#     mlflow_model_deployer_component=MLFlowModelDeployer.get_active_model_deployer()
    
#     existing_services=mlflow_model_deployer_component.find_model_server(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         model_name=model_name,
#         running=running,
#     ) 
    
#     if not existing_services:
#          raise RuntimeError(
#             f"No MLflow prediction service deployed by the "
#             f"{pipeline_step_name} step in the {pipeline_name} "
#             f"pipeline for the '{model_name}' model is currently "
#             f"running."
#         )
#     print(existing_services)
#     print(type(existing_services))
#     return existing_services[0]

# @step()
# def predictor(service: MLFlowDeploymentService,
#               data: np.ndarray)->np.ndarray:
    
#     service.start(timeout=60)
#     data=json.loads(data)
#     data.pop("columns")
#     data.pop("index")
    
#     columns_for_df=[
#         "payment_sequential",
#         "payment_installments",
#         "payment_value",
#         "price",
#         "freight_value",
#         "product_name_lenght",
#         "product_description_lenght",
#         "product_photos_qty",
#         "product_weight_g",
#         "product_length_cm",
#         "product_height_cm",
#         "product_width_cm",
#     ]
#     df = pd.DataFrame(data["data"], columns=columns_for_df)
#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#     data = np.array(json_list)
#     prediction = service.predict(data)
#     return prediction


# @pipeline(enable_cache=True, settings={"docker":docker_settings})
# def continuous_deployment_pipeline(
#     data_path: str,
#     min_accuracy: float=0.5,
#     workers: int=1,
#     timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT, 
# ):
#     df=ingest_df(data_path=data_path)
#     X_train, X_test, y_train, y_test=clean_df(df)
#     model=train_model( X_train, X_test, y_train, y_test)
#     r2_score, rmse=evaluate_model(model, X_test, y_test)
#     deployment_decision=deployment_trigger(r2_score)
#     mlflow_model_deployer_step(
#         model=model,
#         deploy_decision=deployment_decision,
#         workers=workers,
#         timeout=timeout,
#         )

# @pipeline(enable_cache=False, settings={"docker":docker_settings})
# def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    
#     batch_data=dynamic_importer()
#     model_deployment_service=prediction_service_loader(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         running=False,
#     )
#     predictor(service=model_deployment_service, data=batch_data)

import json
import os
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters

# Ensure Docker settings include MLFlow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()  # Ensure this function is properly defined in your utils module
    return data

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.5
    
@step()
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True
    model_name: str = "model"

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step()
def predictor(service: MLFlowDeploymentService, data: np.ndarray) -> np.ndarray:
    service.start(timeout=60)
    data = json.loads(data)
    data.pop("columns", None)
    data.pop("index", None)
    
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)
    prediction = service.predict(data_array)
    return prediction

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)  # Ensure this function is properly defined in your steps module
    X_train, X_test, y_train, y_test = clean_df(df)  # Ensure this function is properly defined in your steps module
    model = train_model(X_train, X_test, y_train, y_test)  # Ensure this function is properly defined in your steps module
    r2_score, rmse = evaluate_model(model, X_test, y_test)  # Ensure this function is properly defined in your steps module
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
