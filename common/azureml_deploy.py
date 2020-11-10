import os

import azureml.core
import mlflow
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
from dotenv import load_dotenv
from loguru import logger

from common.config import PRJ_DIR


def get_workspace():
    ws = Workspace.from_config()
    logger.debug("AzureML SDK version: {}", azureml.core.VERSION)
    logger.debug("MLflow version: {}", mlflow.version.VERSION)
    logger.info(
        f"""
{ws.name}
{ws.resource_group}
{ws.location}
{ws.subscription_id}
"""
    )
    return ws


def build_image(model_uri, exp_name: str, ws: Workspace):
    logger.info("model {}", model_uri)
    model_image, _ = mlflow.azureml.build_image(
        model_uri=model_uri,
        workspace=ws,
        model_name=exp_name,
        image_name=exp_name,
        synchronous=True,
    )
    return model_image


def deploy_image(model_image, exp_name, ws: Workspace):
    logger.info("model_image {}", model_image)
    websvc_deploy_cfg = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)
    websvc = Webservice.deploy_from_image(
        image=model_image,
        name=exp_name,
        deployment_config=websvc_deploy_cfg,
        workspace=ws,
        overwrite=True,
    )
    websvc.wait_for_deployment()
    logger.info(websvc.scoring_uri)
    return websvc


def main():
    load_dotenv(dotenv_path=PRJ_DIR / ".trained.env", override=True)
    model_uri = os.getenv("MLFLOW_MODEL")
    exp_name = os.getenv("EXPERIMENT", "azureml-mlflow")

    ws = get_workspace()
    model_image = build_image(model_uri, exp_name, ws)
    websvc = deploy_image(model_image, exp_name, ws)
    (PRJ_DIR / ".deployed.env").write_text(f"SCORING_URI={websvc.scoring_uri}")
    return websvc.scoring_uri


if __name__ == "__main__":
    main()
