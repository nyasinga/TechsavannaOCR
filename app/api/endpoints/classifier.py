import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import numpy as np
import cv2

from app.schemas.classifier import (
    ClassificationRequest,
    ClassificationResponse,
    ClassScore,
    ClassificationMode,
)
from app.services.classifier_service import classifier_service

router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    mode: str = Form("image"),
    top_k: int = Form(3),
    language: str = Form("eng"),
    text: Optional[str] = Form(None),
):
    start = time.time()

    try:
        # Validate mode
        try:
            use_mode = ClassificationMode(mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use 'image' or 'text'.")

        # Prepare input per mode
        if use_mode == ClassificationMode.image:
            if file is None:
                raise HTTPException(status_code=400, detail="File is required for image mode")
            # Read bytes and decode image
            img_bytes = await file.read()
            if not img_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")
            label, conf, top = classifier_service.classify_image(image=image, top_k=top_k, language=language)
        else:
            # text mode
            if not text:
                # If text not given but file provided, try to read file as text
                if file is not None:
                    content = (await file.read()).decode(errors="ignore")
                    text = content
            if not text:
                raise HTTPException(status_code=400, detail="No text provided for text mode")
            label, conf, top = classifier_service.classify_text(text=text, top_k=top_k)

        needs_review = classifier_service.needs_manual_review(conf)
        processing_time = time.time() - start

        response = ClassificationResponse(
            label=label,
            confidence=float(conf),
            top_k=[ClassScore(label=l, score=float(s)) for l, s in top],
            mode=use_mode,
            needs_review=needs_review,
            processing_time=processing_time,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
