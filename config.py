from pathlib import Path

# Root Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
DATASET_PATH = RAW_DATA_DIR / "Bank_dataset.xlsx"

# Models
MODEL_DIR = BASE_DIR / "models"

# Reports
REPORT_DIR = BASE_DIR / "reports"

# Assets
ASSET_DIR = BASE_DIR / "assets" 