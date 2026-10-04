from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
PUBLIC_DATA_DIR = DATASET_DIR / "public"
PREPARED_DATA_DIR = DATASET_DIR / "prepared"
MODELS_DIR = ROOT_DIR / "models"
RUNTIME_DIR = ROOT_DIR / "runtime"

DEFAULT_DATASET_ID = os.getenv("LUCID_DATASET_ID", "nm000113")
DEFAULT_DATASET_VERSION = os.getenv("LUCID_DATASET_VERSION", "v1.0.0")
DEFAULT_PREPARED_PATH = Path(os.getenv("LUCID_PREPARED_PATH", str(PREPARED_DATA_DIR / f"{DEFAULT_DATASET_ID}_words.npz")))
DEFAULT_MODEL_PATH = Path(os.getenv("LUCID_MODEL_PATH", str(MODELS_DIR / "eegnet_words.pt")))
DB_PATH = Path(os.getenv("LUCID_DB_PATH", str(RUNTIME_DIR / "lucid.sqlite3")))

for path in (PUBLIC_DATA_DIR, PREPARED_DATA_DIR, MODELS_DIR, RUNTIME_DIR):
    path.mkdir(parents=True, exist_ok=True)
