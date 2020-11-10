import joblib
import mlflow
import pandas as pd
from loguru import logger
from preproc_base import Preproc


def unpack_model_zip(ml_model_zip: str) -> str:
    import tempfile
    from shutil import unpack_archive

    ml_model_dir = tempfile.mkdtemp()
    unpack_archive(ml_model_zip, extract_dir=ml_model_dir)
    return ml_model_dir


class AutoGluonPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        from autogluon.tabular import TabularPrediction as task

        logger.info(f"artifacts {context.artifacts}")
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])

        ml_model = unpack_model_zip(context.artifacts["ml_model"])
        self.ml_model = task.load(ml_model)

        # Keeping models in memory. BUG: loading deleted models
        # self.ml_model.persist_models()

    def predict(self, context, input: pd.DataFrame) -> pd.DataFrame:
        input = self.pre_model.transform(input)
        proba = self.ml_model.predict_proba(input)
        return pd.DataFrame({"proba": proba})
