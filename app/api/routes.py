from fastapi import APIRouter
from .endpoints import ocr as ocr_endpoints
from .endpoints import classifier as classifier_endpoints

api_router = APIRouter()
api_router.include_router(ocr_endpoints.router, prefix="/ocr", tags=["OCR"])
api_router.include_router(classifier_endpoints.router, prefix="/classifier", tags=["Classifier"])
