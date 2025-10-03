from pydantic import BaseModel, Field, conlist, confloat
from typing import List, Dict, Optional, Literal, Union, Any
from enum import Enum
from datetime import datetime

class ClassificationMode(str, Enum):
    """Supported classification modes"""
    IMAGE = "image"
    TEXT = "text"

class DocumentType(str, Enum):
    """Supported document types for classification"""
    KRA_PIN_CERTIFICATE = "kra_pin_certificate"
    NATIONAL_ID = "national_id"
    CERTIFICATE_OF_REGISTRATION = "certificate_of_registration"
    BUSINESS_PERMIT = "business_permit"
    CR12 = "cr12"
    SERVICE_APPLICATION_FORM = "service_application_form"
    UNKNOWN = "unknown"

class ClassScore(BaseModel):
    """Class score with confidence"""
    label: str = Field(..., description="Class label")
    score: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence score between 0 and 1")

class ClassificationRequest(BaseModel):
    """Request model for document classification"""
    mode: ClassificationMode = Field(
        default=ClassificationMode.IMAGE, 
        description="Classification mode: 'image' for image files or 'text' for direct text input"
    )
    text: Optional[str] = Field(
        default=None, 
        description="Text content to classify (required when mode='text')"
    )
    top_k: int = Field(
        default=3, 
        ge=1, 
        le=5, 
        description="Number of top predictions to return"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0-1) for a prediction to be considered valid"
    )

class ClassificationResponse(BaseModel):
    """Response model for document classification"""
    document_type: DocumentType = Field(
        ...,
        description="Predicted document type"
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Confidence score of the prediction (0-1)"
    )
    predictions: List[ClassScore] = Field(
        ...,
        description="List of top predictions with confidence scores"
    )
    needs_review: bool = Field(
        ...,
        description="True if the confidence is below the threshold or prediction is uncertain"
    )
    processing_time: float = Field(
        ...,
        gt=0,
        description="Time taken to process the classification in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the classification was performed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the classification"
    )

    class Config:
        schema_extra = {
            "example": {
                "document_type": "kra_pin_certificate",
                "confidence": 0.92,
                "predictions": [
                    {"label": "kra_pin_certificate", "score": 0.92},
                    {"label": "national_id", "score": 0.07},
                    {"label": "unknown", "score": 0.01}
                ],
                "needs_review": False,
                "processing_time": 0.15,
                "timestamp": "2025-10-03T10:00:00.000000",
                "metadata": {
                    "version": "1.0.0",
                    "model": "document-classifier-v1"
                }
            }
        }
