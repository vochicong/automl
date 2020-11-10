import h2o
import joblib
import mlflow
import pandas as pd
from loguru import logger
from preproc_base import Preproc


class H2OPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        h2o.init()
        logger.info(f"artifacts {context.artifacts}")
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])
        self.ml_model = h2o.load_model(context.artifacts["ml_model"])

    def predict(self, context, df_input: pd.DataFrame) -> pd.DataFrame:
        hf_input = h2o.H2OFrame(self.pre_model.transform(df_input))
        output: h2o.H2OFrame = self.ml_model.predict(hf_input)
        proba = output.as_data_frame()["p0"].values
        return pd.DataFrame({"proba": proba})
