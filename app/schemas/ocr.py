from pydantic import BaseModel, Field, HttpUrl, confloat, conlist, validator
from typing import List, Optional, Dict, Any, Literal, Union
from enum import Enum
from datetime import datetime
from typing_extensions import Annotated

# Constants for validation
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0
MAX_WORD_LENGTH = 200

class OCREngine(str, Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    PADDLE = "paddle"

class WordBox(BaseModel):
    """Represents a single word with its position and confidence"""
    text: str = Field(..., description="The extracted text of the word")
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ..., 
        description="Confidence score between 0 and 1"
    )
    position: conlist(int, min_items=4, max_items=4) = Field(
        ...,
        description="Bounding box coordinates [x1, y1, x2, y2]"
    )
    polygon: Optional[conlist(int, min_items=8, max_items=8)] = Field(
        None,
        description="Optional polygon coordinates [x1,y1,x2,y2,x3,y3,x4,y4] for more precise text localization"
    )
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v) > MAX_WORD_LENGTH:
            raise ValueError(f'Word text exceeds maximum length of {MAX_WORD_LENGTH} characters')
        return v.strip()
    
    @validator('position')
    def validate_position(cls, v):
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid bounding box coordinates')
        return v

class OCRRequest(BaseModel):
    """Request model for OCR text extraction"""
    language: str = Field(
        default="eng",
        description="Language code for OCR (e.g., 'eng', 'fra', 'spa')",
        min_length=2,
        max_length=10,
        example="eng"
    )
    
    engine: OCREngine = Field(
        default=OCREngine.TESSERACT,
        description="OCR engine to use for text extraction",
        example="tesseract"
    )
    
    preprocess: bool = Field(
        default=True,
        description="Whether to apply image preprocessing for better OCR results",
        example=True
    )
    
    include_word_boxes: bool = Field(
        default=True,
        description="Whether to include word-level bounding boxes in the response",
        example=True
    )
    
    class Config:
        schema_extra = {
            "example": {
                "language": "eng",
                "engine": "tesseract",
                "preprocess": True,
                "include_word_boxes": True
            }
        }

class BusinessPermitResponse(BaseModel):
    """Response model for Business Permit extraction"""
    # Document Information
    document_type: str = Field("business_permit", description="Type of document")
    permit_number: Optional[str] = Field(None, description="Business permit number")
    receipt_number: Optional[str] = Field(None, description="Payment receipt number")
    
    # Business Information
    business_name: Optional[str] = Field(None, description="Registered business name")
    trading_name: Optional[str] = Field(None, description="Trading name (if different from registered name)")
    business_activity: Optional[str] = Field(None, description="Primary business activity")
    business_category: Optional[str] = Field(None, description="Category of business")
    
    # Location Information
    physical_address: Optional[Dict[str, str]] = Field(
        None,
        description="Structured address information"
    )
    ward: Optional[str] = Field(None, description="Ward where business is located")
    sub_county: Optional[str] = Field(None, description="Sub-county where business is located")
    county: Optional[str] = Field(None, description="County where business is registered")
    
    # Permit Details
    permit_type: Optional[str] = Field(None, description="Type of business permit")
    date_issued: Optional[str] = Field(None, description="Date when permit was issued (YYYY-MM-DD)")
    expiry_date: Optional[str] = Field(None, description="Date when permit expires (YYYY-MM-DD)")
    
    # Financial Information
    permit_fee: Optional[float] = Field(None, description="Amount paid for the permit")
    payment_date: Optional[str] = Field(None, description="Date of payment (YYYY-MM-DD)")
    
    # Contact Information
    owner_name: Optional[str] = Field(None, description="Name of business owner")
    phone_number: Optional[str] = Field(None, description="Contact phone number")
    email: Optional[str] = Field(None, description="Contact email address")
    
    # System Information
    processing_time: float = Field(..., description="Time taken to process the document in seconds", gt=0)
    engine_used: str = Field(..., description="The OCR engine used for text extraction")
    confidence: Optional[float] = Field(None, description="Confidence score of the extraction (0-1)")
    raw_text: Optional[str] = Field(None, description="Raw OCR text (useful for debugging)")

    class Config:
        schema_extra = {
            "example": {
                "document_type": "business_permit",
                "permit_number": "BP/2023/12345",
                "receipt_number": "RCPT/2023/67890",
                "business_name": "JAMBO TRADERS LIMITED",
                "trading_name": "JAMBO SUPERMARKET",
                "business_activity": "RETAIL TRADING",
                "business_category": "GENERAL SHOP",
                "physical_address": {
                    "building": "PLAZA BUILDING",
                    "street": "MAIN STREET",
                    "town": "NAIROBI"
                },
                "ward": "CENTRAL WARD",
                "sub_county": "WESTLANDS",
                "county": "NAIROBI CITY",
                "permit_type": "ANNUAL BUSINESS PERMIT",
                "date_issued": "2023-01-15",
                "expiry_date": "2023-12-31",
                "permit_fee": 15000.00,
                "payment_date": "2023-01-10",
                "owner_name": "JOHN DOE MWANGI",
                "phone_number": "+254712345678",
                "email": "info@jambotraders.co.ke",
                "processing_time": 1.25,
                "engine_used": "tesseract",
                "confidence": 0.92,
                "raw_text": "..."
            }
        }

class KRAPINResponse(BaseModel):
    """Response model for KRA PIN certificate extraction"""
    taxpayer_name: Optional[str] = Field(None, description="Tax Payer Name as printed on the certificate")
    email_address: Optional[str] = Field(None, description="Email Address on the certificate")
    personal_identification_number: Optional[str] = Field(None, description="KRA Personal Identification Number (PIN)")
    certificate_date: Optional[str] = Field(None, description="Date when the certificate was issued (YYYY-MM-DD format)")
    registered_address: Optional[Dict[str, str]] = Field(
        None,
        description="Structured address information containing line1, line2, and city"
    )
    county: Optional[str] = Field(None, description="County of residence")
    district: Optional[str] = Field(None, description="District of residence")
    postal_code: Optional[str] = Field(None, description="Postal code")
    tax_obligations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of tax obligations with type, effective_from, and status"
    )
    processing_time: float = Field(..., description="Time taken to process the document in seconds", gt=0)
    engine_used: str = Field(..., description="The OCR engine used for text extraction")
    raw_text: Optional[str] = Field(None, description="Raw OCR text (useful for debugging)")

    class Config:
        schema_extra = {
            "example": {
                "taxpayer_name": "DUKE NYARAKE NYASING'A",
                "email_address": "NYASINGADUKE@GMAIL.COM",
                "personal_identification_number": "A007341474P",
                "certificate_date": "2023-05-15",
                "registered_address": {
                    "line1": "123 MAIN STREET",
                    "line2": "P.O. BOX 12345",
                    "city": "NAIROBI"
                },
                "county": "NAIROBI",
                "district": "WESTLANDS",
                "postal_code": "00100",
                "tax_obligations": [
                    {
                        "type": "VAT",
                        "effective_from": "01/01/2020",
                        "status": "Active"
                    },
                    {
                        "type": "PAYE",
                        "effective_from": "01/01/2020",
                        "status": "Active"
                    }
                ],
                "processing_time": 1.25,
                "engine_used": "tesseract",
                "raw_text": "..."
            }
        }


class OCRResponse(BaseModel):
    """Response model for OCR text extraction"""
    text: str = Field(..., description="The extracted text from the image")
    
    confidence: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None,
        description="Average confidence score of the extracted text (0-1)"
    )
    
    engine: str = Field(..., description="The OCR engine used for text extraction")
    language: str = Field(..., description="Language code used for OCR")
    
    processing_time: float = Field(
        ...,
        description="Time taken to process the image in seconds",
        gt=0
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the OCR processing was completed"
    )
    
    word_boxes: Optional[List[WordBox]] = Field(
        None,
        description="List of detected words with their positions and confidence scores"
    )
    
    page_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional page-level metrics and statistics"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "text": "Sample extracted text",
                "confidence": 0.95,
                "engine": "tesseract",
                "language": "eng",
                "processing_time": 1.23,
                "timestamp": "2023-06-15T12:34:56.789Z",
                "word_boxes": [
                    {
                        "text": "Sample",
                        "confidence": 0.96,
                        "position": [10, 20, 100, 30],
                        "polygon": [10, 20, 100, 20, 100, 30, 10, 30]
                    },
                    {
                        "text": "extracted",
                        "confidence": 0.94,
                        "position": [110, 20, 200, 30],
                        "polygon": [110, 20, 200, 20, 200, 30, 110, 30]
                    },
                    {
                        "text": "text",
                        "confidence": 0.95,
                        "position": [210, 20, 250, 30],
                        "polygon": [210, 20, 250, 20, 250, 30, 210, 30]
                    }
                ],
                "page_metrics": {
                    "width": 800,
                    "height": 600,
                    "word_count": 3
                }
            }
        }

class IDDocumentResponse(BaseModel):
    """Response model for ID document name extraction"""
    full_name: str = Field(
        ...,
        description="The most probable full name extracted from the ID document"
    )
    all_matches: List[str] = Field(
        ...,
        description="All potential name matches found in the document"
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the document in seconds",
        gt=0
    )
    engine_used: str = Field(
        ...,
        description="The OCR engine(s) used for text extraction"
    )

    class Config:
        schema_extra = {
            "example": {
                "full_name": "John Doe",
                "all_matches": ["John Doe", "Doe John"],
                "processing_time": 1.23,
                "engine_used": "Tesseract + PaddleOCR"
            }
        }

class LanguageSupport(BaseModel):
    """Model for language support information"""
    engine: str = Field(..., description="OCR engine name")
    languages: Dict[str, str] = Field(
        ...,
        description="Dictionary of language codes and their display names"
    )

    class Config:
        schema_extra = {
            "example": {
                "engine": "tesseract",
                "languages": {
                    "eng": "English"
                }
            }
        }
