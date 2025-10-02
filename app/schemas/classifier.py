from pydantic import BaseModel, Field, conlist, confloat
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class ClassificationMode(str, Enum):
    image = "image"
    text = "text"

class ClassScore(BaseModel):
    label: str = Field(..., description="Class label")
    score: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence for the label (0-1)")

class ClassificationRequest(BaseModel):
    mode: ClassificationMode = Field(default=ClassificationMode.image, description="Classification mode: image or text")
    text: Optional[str] = Field(default=None, description="Optional OCR text to classify directly if mode=text")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top classes to return")

class ClassificationResponse(BaseModel):
    label: str = Field(..., description="Predicted document class label")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence of the predicted label (0-1)")
    top_k: List[ClassScore] = Field(..., description="Top-K class scores")
    mode: ClassificationMode = Field(..., description="Mode used for classification")
    needs_review: bool = Field(..., description="True if confidence below review threshold")
    processing_time: float = Field(..., gt=0, description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
