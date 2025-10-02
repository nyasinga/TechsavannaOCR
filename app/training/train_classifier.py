import argparse
import os
from pathlib import Path
from typing import List, Tuple
import sys
import numpy as np

# Ensure package imports work when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.services.tesseract_service import tesseract_ocr
from app.services.paddle_service import paddle_ocr

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def ocr_image(image_path: Path, engine: str = "tesseract", language: str = "eng") -> str:
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if engine == "paddle":
        res = paddle_ocr.extract_text(image=img, language=language, preprocess=True)
    else:
        res = tesseract_ocr.extract_text(image=img, language=language, preprocess=True)
    return res.get("text", "")


def load_dataset(data_dir: Path, engine: str, language: str) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for f in label_dir.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                try:
                    text = ocr_image(f, engine=engine, language=language)
                    if text and text.strip():
                        X.append(text)
                        y.append(label)
                except Exception as e:
                    print(f"[WARN] Skipping {f}: {e}")
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=100000,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                multi_class="auto",
            ),
        ),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train document classifier (OCR text -> TFIDF + LogisticRegression)")
    parser.add_argument("--data_dir", type=str, default=str(Path("data/train").absolute()), help="Training data directory: subfolders per label")
    parser.add_argument("--engine", type=str, default="tesseract", choices=["tesseract", "paddle"], help="OCR engine to use for training")
    parser.add_argument("--language", type=str, default="eng", help="OCR language code")
    parser.add_argument("--model_out", type=str, default=settings.CLASSIFIER_MODEL_PATH, help="Path to save trained model (joblib)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Loading dataset from: {data_dir}")
    print(f"OCR engine: {args.engine}, language: {args.language}")

    X, y = load_dataset(data_dir, engine=args.engine, language=args.language)
    if not X:
        raise SystemExit("No training samples found. Place images under data_dir/<Label>/*.jpg")

    print(f"Samples: {len(X)} | Classes: {sorted(set(y))}")

    pipe = build_pipeline()
    pipe.fit(X, y)

    # Evaluate on training set (with only a few samples this will be near 100%)
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training accuracy: {acc:.4f}")
    try:
        print(classification_report(y, y_pred))
    except Exception:
        pass

    # Ensure output dir
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, str(model_path))
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
