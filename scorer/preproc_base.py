from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


class Preproc:
    SCALE_COLS = ["Age", "Fare"]

    def __init__(self) -> None:
        pass

    def fit(self, df) -> None:
        logger.info(f"Original Columns {df.columns}")
        logger.info(f"MinMaxScaler fit {self.SCALE_COLS}")
        self.scaler = MinMaxScaler().fit(df[self.SCALE_COLS])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop("Name", axis="columns")
        logger.debug(f"MinMaxScaler transform {self.SCALE_COLS}")
        df[self.SCALE_COLS] = self.scaler.transform(df[self.SCALE_COLS])
        logger.debug(f"df head\n{df.head()}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save_model(self, model_path: Union[Path, str]) -> str:
        joblib.dump(self, model_path)
        return model_path
