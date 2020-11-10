from typing import List

import mlflow
from loguru import logger

from common.config import PRJ_DIR


def log_model(
    pre_model, ml_model, predictor_model, predictor_code: List[str], conda_env: str
):
    # %% MLflowで学習済みの前処理・モデルを保存
    artifacts = dict(
        pre_model=pre_model,
        #
        ml_model=ml_model,
    )

    artifact_path = "automl"
    logger.debug(artifacts)
    logger.debug(conda_env)
    mlflow_model_info = dict(
        artifact_path=artifact_path,
        python_model=predictor_model,
        artifacts=artifacts,
        conda_env=conda_env,
        code_path=predictor_code,
    )
    mlflow.pyfunc.log_model(**mlflow_model_info)
    logger.info(f"mlflow log_model {mlflow_model_info}")
    mlflow_model = mlflow.get_artifact_uri(artifact_path)
    prefix = "file://"
    if mlflow_model.startswith(prefix):
        mlflow_model = mlflow_model[len(prefix) :]

    mlflow.end_run()

    (PRJ_DIR / ".trained.env").write_text(
        f"""
MLFLOW_MODEL={mlflow_model}
""".strip()
    )
