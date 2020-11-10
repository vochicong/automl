import os
from pathlib import Path

DATA_URI: str = (
    "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
)
Y_TARGET = "Survived"

MAX_TRAIN_SECS = 10

PRJ_DIR = Path(__file__).parents[1]
DATA_DIR = PRJ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR = DATA_DIR / "test"
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_CSV = os.getenv("TEST_CSV", str(TEST_DATA_DIR / "dftest.csv"))
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
TMP_DIR = PRJ_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)
MODEL_DIR = PRJ_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
