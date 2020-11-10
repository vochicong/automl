from pathlib import Path

import pandas as pd
from loguru import logger
from preproc_base import Preproc
from sklearn.model_selection import train_test_split

from common.config import DATA_URI, MODEL_DIR, PROCESSED_DATA_DIR, TEST_CSV


def preprocess(csv=DATA_URI):
    # %% 実験データのダウンロード
    data = pd.read_csv(csv)
    logger.debug("Downloaded data")

    # %% 学習・テストデータの分割
    dftrain: pd.DataFrame
    dftest: pd.DataFrame
    dftrain, dftest = train_test_split(data)
    logger.info(f"train_test_split {data.shape} -> {dftrain.shape} + {dftest.shape}")

    # Save test dataframe for later tests
    Path("../data").mkdir(exist_ok=True)
    dftest.to_csv(TEST_CSV, index=False)

    # %% データ前処理
    prep = Preproc()
    train = prep.fit_transform(dftrain)
    pre_model = prep.save_model(MODEL_DIR / "prep.model")
    test = prep.transform(dftest)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    return train, test, pre_model


def read_processed_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    pre_model = str(MODEL_DIR / "prep.model")
    return train, test, pre_model
