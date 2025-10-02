import time
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
import cv2

from ..core.config import settings
from .tesseract_service import tesseract_ocr
import os
from pathlib import Path

try:
    import joblib  # For loading trained model
except Exception:
    joblib = None

class DocumentClassifierService:
    def __init__(self):
        # Labels configured via settings
        self.labels: List[str] = settings.CLASSIFIER_LABELS
        # Basic keyword patterns per label (can be extended or replaced by ML model later)
        self.keyword_map: Dict[str, List[re.Pattern]] = self._build_keyword_map()
        self.review_threshold: float = settings.CLASSIFIER_REVIEW_THRESHOLD
        # Model members
        self._model = None
        self._model_loaded = False

    def _build_keyword_map(self) -> Dict[str, List[re.Pattern]]:
        patterns: Dict[str, List[str]] = {
            "Certificate of Registration": [
                r"\bcertificate\b",
                r"\bregistration\b",
                r"\bcompany\b",
                r"\bincorporation\b",
                r"\bregistered\b",
            ],
            "Pin Certificate": [
                r"\btax\b",
                r"\bpin\b",
                r"\bpin\s*certificate\b",
                r"\bpersonal\s+identification\s+number\b",
                r"\bkenya\s+revenue\s+authority\b",
                r"\bkra\b",
                r"\bregistration\b",
                r"\bdomestic\s+tax\b",
                r"\btaxpayer\s+information\b",
                r"\btax\s+obligation\(s\)\b|\btax\s+obligations\b",
                r"\bcertificate\s+date\b",
            ],
            "Business Permit": [
                r"\bbusiness\b",
                r"\bpermit\b",
                r"\blicence|license\b",
                r"\bcounty\b",
            ],
            "CR12": [
                r"\bcr\s*12\b",
                r"\bcompany\s+registry\b",
                r"\bshareholder\b",
                r"\bdirector\b",
            ],
            "Service Application Forms": [
                r"\bapplication\b",
                r"\bform\b",
                r"\bapplicant\b",
                r"\bsignature\b",
            ],
            "ID Cards": [
                r"\bnational\s+id\b",
                r"\bidentity\s+card\b",
                r"\bid\s*no\.?\b",
                r"\bid\s*number\b",
                r"\brepublic\s+of\s+kenya\b",
                r"\bjamhuri\s+ya\s+kenya\b",
                r"\bfull\s+names\b",
                r"\bdate\s+of\s+birth\b",
                r"\bsex\b",
                r"\bholder'?s\s+sign\b",
            ],
            "Other": [
                r"\bgovernment\b",
                r"\bministry\b",
                r"\bdepartment\b",
                r"\bauthority\b",
            ],
        }
        # Compile regex
        compiled: Dict[str, List[re.Pattern]] = {}
        for label in self.labels:
            kws = patterns.get(label, [])
            compiled[label] = [re.compile(pat, re.IGNORECASE) for pat in kws]
        return compiled

    def _score_text(self, text: str) -> Dict[str, float]:
        # Basic keyword scoring; 
        scores: Dict[str, float] = {label: 0.0 for label in self.labels}
        # Normalize whitespace
        t = re.sub(r"\s+", " ", text or "").strip()
        if not t:
            return scores
        # Score per label
        for label, patterns in self.keyword_map.items():
            score = 0.0
            for pat in patterns:
                # Count matches
                matches = list(pat.finditer(t))
                if matches:
                    score += len(matches)
            scores[label] = score
        # Targeted boosts for strong indicators
        try:
            # Pin Certificate strong signals
            if re.search(r"\bpin\s*certificate\b", t, re.IGNORECASE):
                scores["Pin Certificate"] = scores.get("Pin Certificate", 0.0) + 3.0
            if re.search(r"\bkenya\s+revenue\s+authority\b|\bkra\b", t, re.IGNORECASE):
                scores["Pin Certificate"] = scores.get("Pin Certificate", 0.0) + 2.0
            # PIN format e.g., A007341474P or A 007 341 474 P
            if re.search(r"\b[A-Z]\s*\d{3}\s*\d{3}\s*\d{3}\s*[A-Z]\b", t, re.IGNORECASE):
                scores["Pin Certificate"] = scores.get("Pin Certificate", 0.0) + 3.0
            if re.search(r"\btaxpayer\s+information\b|\btax\s+obligation", t, re.IGNORECASE):
                scores["Pin Certificate"] = scores.get("Pin Certificate", 0.0) + 2.0

            # ID Cards strong signals
            if re.search(r"\brepublic\s+of\s+kenya\b|\bjamhuri\s+ya\s+kenya\b", t, re.IGNORECASE):
                scores["ID Cards"] = scores.get("ID Cards", 0.0) + 2.0
            if re.search(r"\bid\s*number\b|\bidentity\s+card\b|\bnational\s+id\b", t, re.IGNORECASE):
                scores["ID Cards"] = scores.get("ID Cards", 0.0) + 2.0
            # Common paired cues on Kenyan IDs
            id_pair_hits = 0
            if re.search(r"\bfull\s+names\b", t, re.IGNORECASE):
                id_pair_hits += 1
            if re.search(r"\bdate\s+of\s+birth\b", t, re.IGNORECASE):
                id_pair_hits += 1
            if id_pair_hits >= 2:
                scores["ID Cards"] = scores.get("ID Cards", 0.0) + 2.0
            if re.search(r"holder'?s\s+sign", t, re.IGNORECASE):
                scores["ID Cards"] = scores.get("ID Cards", 0.0) + 1.0
        except Exception:
            # Fail-safe: ignore boost errors
            pass
        # If all zero, keep zeros. Else normalize to 0..1 by dividing by max
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            for k in scores.keys():
                scores[k] = scores[k] / max_score
        return scores

    def _ensure_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        if not settings.CLASSIFIER_USE_MODEL:
            return
        try:
            model_path = Path(settings.CLASSIFIER_MODEL_PATH)
            if joblib and model_path.exists():
                self._model = joblib.load(str(model_path))
        except Exception:
            # Leave model as None if loading fails
            self._model = None

    def classify_text(self, text: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
        # Try ML model first if available
        self._ensure_model()
        if self._model is not None:
            try:
                # Predict probabilities
                proba = self._model.predict_proba([text or ""])  # shape (1, n_classes)
                classes = list(getattr(self._model, "classes_", self.labels))
                pairs = list(zip(classes, proba[0].tolist()))
                pairs.sort(key=lambda x: x[1], reverse=True)
                top_pairs = pairs[:top_k]
                label, confidence = (top_pairs[0][0], float(top_pairs[0][1])) if top_pairs else (self.labels[0], 0.0)
                return label, confidence, top_pairs
            except Exception:
                # Fall back to rules
                pass
        # Rule-based fallback
        scores = self._score_text(text)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_scores[:top_k]
        label, confidence = (top[0][0], top[0][1]) if top else (self.labels[0], 0.0)
        return label, confidence, top

    def classify_image(self, image: np.ndarray, top_k: int = 3, language: str = "eng") -> Tuple[str, float, List[Tuple[str, float]]]:
        # Use OCR to extract text then classify text
        ocr_result = tesseract_ocr.extract_text(image=image, language=language, preprocess=True)
        text = ocr_result.get("text", "")
        return self.classify_text(text=text, top_k=top_k)

    def needs_manual_review(self, confidence: float) -> bool:
        return confidence < self.review_threshold

# Singleton
classifier_service = DocumentClassifierService()
