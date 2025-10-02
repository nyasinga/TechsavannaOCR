from pydantic import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = str(Path("uploads").absolute())
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
    
    # Tesseract settings
    TESSERACT_CMD: str = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update this path if needed
    
    # PaddleOCR settings
    USE_GPU: bool = False  # Set to True if you have CUDA GPU
    
    # Classifier settings
    CLASSIFIER_LABELS: list = [
        "Certificate of Registration",
        "Pin Certificate",
        "Business Permit",
        "CR12",
        "Service Application Forms",
        "ID Cards",
        "Other",
    ]
    # Flag for manual review when confidence below this threshold
    CLASSIFIER_REVIEW_THRESHOLD: float = 0.55
    # Path to trained text classifier (TF-IDF + LogisticRegression) saved with joblib
    CLASSIFIER_MODEL_PATH: str = str(Path("models/doc_text_clf.joblib").absolute())
    # Whether to prefer trained model over rules when present
    CLASSIFIER_USE_MODEL: bool = True
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:3000",  # React default port
        "http://localhost:8000",  # FastAPI default port
    ]
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
