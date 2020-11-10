# %%
import tempfile
import warnings

from h2o_mlflow_predictor import H2OPredictor

from common.mlflow_util import log_model
from common.preprocess import read_processed_data

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import h2o

import mlflow
import mlflow.h2o
import mlflow.pyfunc
from h2o import H2OFrame
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from loguru import logger

from common.config import MAX_TRAIN_SECS, MODEL_DIR, PRJ_DIR, Y_TARGET


def h2o_fit(train: H2OFrame, test: H2OFrame) -> str:
    x = train.columns
    y = Y_TARGET
    x.remove(y)
    logger.info(f"x={x}, y={y}")

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    aml = H2OAutoML(max_runtime_secs=MAX_TRAIN_SECS)
    aml.train(x=x, y=y, training_frame=train)

    # View the AutoML Leaderboard
    logger.info(aml.leaderboard)

    val_auc = aml.leader.model_performance(xval=True).auc()
    test_auc = aml.leader.model_performance(test).auc()

    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)
    mlflow.log_param("algos", "H2OAutoML")
    mlflow.log_param("max_secs", MAX_TRAIN_SECS)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    # mlflow.h2o.log_model(aml.leader, "model")

    model_path = tempfile.mkdtemp(dir=MODEL_DIR)
    model_path = h2o.save_model(model=aml.leader, path=model_path)
    logger.debug(f"h2o.save_model to {model_path} DONE")
    return model_path


def train_h2o():
    """A pipeline to
    - Read CSV and preprocess data
    - Train using H2OAutoML
    - Save trained models
    """
    mlflow.start_run()
    train, test, pre_model = read_processed_data()

    # %% 機械学習 AutoML
    h2o.init()
    ml_model = h2o_fit(H2OFrame(train), H2OFrame(test))

    log_model(
        pre_model,
        ml_model,
        H2OPredictor(),
        [
            str(PRJ_DIR / "scorer/h2o_mlflow_predictor.py"),
            str(PRJ_DIR / "scorer/preproc_base.py"),
        ],
        str(PRJ_DIR / "h2o_mlflow/conda.yml"),
    )
