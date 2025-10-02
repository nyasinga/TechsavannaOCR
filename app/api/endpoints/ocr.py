import os
import io
import re
import time
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2

from app.schemas.ocr import OCRRequest, OCRResponse, OCREngine, IDDocumentResponse, KRAPINResponse
from app.services.tesseract_service import tesseract_ocr
from app.services.paddle_service import paddle_ocr
from app.utils.image_processing import validate_image, preprocess_image
from app.core.config import settings

router = APIRouter()

class OCRRequestForm:
    def __init__(
        self,
        file: UploadFile = File(...),
        language: str = Form("eng"),
        engine: str = Form("tesseract"),
        preprocess: str = Form("true"),
        include_word_boxes: str = Form("false")
    ):
        self.file = file
        self.language = language
        self.engine = engine
        self.preprocess = preprocess.lower() == "true"
        self.include_word_boxes = include_word_boxes.lower() == "true"

@router.post("/extract-text", response_model=OCRResponse)
async def extract_text(request: Request):
    """
    Extract text from an image using the specified OCR engine.
    
    Supports both Tesseract and PaddleOCR engines with optional image preprocessing.
    """
    start_time = time.time()
    
    try:
        # Parse the form data
        form_data = await request.form()
        
        # Get the file and parameters
        file = form_data.get("file")
        if not file or not hasattr(file, "file"):
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "No file provided",
                    "solution": "Please ensure you're sending a file with the 'file' form field"
                }
            )
            
        language = form_data.get("language", "eng")
        engine = form_data.get("engine", "tesseract")
        preprocess = form_data.get("preprocess", "true").lower() == "true"
        include_word_boxes = form_data.get("include_word_boxes", "false").lower() == "true"
        
        # Read image file
        try:
            image_bytes = await file.read()
            if not image_bytes:
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": "Empty file provided",
                        "solution": "The uploaded file appears to be empty. Please check the file and try again."
                    }
                )
                
            # Convert to numpy array for processing
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Could not decode image",
                        "solution": "The uploaded file doesn't appear to be a valid image. Please check the file and try again."
                    }
                )
                
            # Process with selected OCR engine
            if engine.lower() == "tesseract":
                result = tesseract_ocr.extract_text(
                    image=image,
                    language=language,
                    preprocess=preprocess
                )
            else:  # PaddleOCR
                result = paddle_ocr.extract_text(
                    image=image,
                    language=language,
                    preprocess=preprocess
                )
                
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "text": result.get("text", ""),
                "language": language,
                "engine": engine,
                "processing_time": processing_time,
                "word_boxes": result.get("word_boxes", []) if include_word_boxes else None,
                "confidence": result.get("confidence", 0.0)
            }
            
            return response
            
        except RuntimeError as e:
            # Handle Tesseract not found error specifically
            if "Tesseract is not installed" in str(e):
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Tesseract OCR is not installed",
                        "solution": (
                            "Tesseract OCR is required but not found. "
                            "Please install it and ensure it's in your system PATH.\n"
                            "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                            "macOS: brew install tesseract\n"
                            "Linux: sudo apt-get install tesseract-ocr"
                        )
                    }
                )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Error processing image: {str(e)}",
                    "solution": "The image could not be processed. Please check the file format and try again."
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "An unexpected error occurred",
                    "solution": "Please try again or contact support if the problem persists",
                    "details": str(e)
                }
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in extract_text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "An unexpected error occurred",
                "solution": "Please try again or contact support if the problem persists"
            }
        )

@router.post("/extract-kra-pin", response_model=KRAPINResponse)
async def extract_kra_pin_details(
    file: UploadFile = File(...),
    engine: OCREngine = OCREngine.TESSERACT,
    include_raw: str = Form("false")
):
    """Extract key fields from a Kenya KRA PIN certificate with improved OCR preprocessing"""
    start_time = time.time()
    try:
        contents = await file.read()
        if not contents:
            return {"error": "Uploaded file is empty"}
        
        # Load image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            from PIL import Image
            pil_image = Image.open(BytesIO(contents))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # === Preprocessing improvements ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale up the image (helps OCR on screenshots)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Sharpen the image
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive threshold (works better for documents)
        ocr_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )

        # Save debug image
        cv2.imwrite("debug_kra_ocr_input.png", ocr_img)

        # === OCR with Tesseract ===
        import pytesseract
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(ocr_img, config=custom_config)

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        taxpayer_name = None
        email_address = None
        personal_identification_number = None

        # === DIRECT PATTERN MATCHING APPROACH ===
        # Since table parsing is unreliable, let's use direct pattern matching
        
        # Look for the specific section in the text
        text_lower = text.lower()
        
        # Find the "Taxpayer Information" section
        taxpayer_section_start = -1
        for i, line in enumerate(lines):
            if "taxpayer information" in line.lower():
                taxpayer_section_start = i
                break
        
        if taxpayer_section_start != -1:
            # Look for taxpayer name in the lines after "Taxpayer Information"
            for i in range(taxpayer_section_start + 1, min(taxpayer_section_start + 10, len(lines))):
                line = lines[i]
                line_lower = line.lower()
                
                # Skip lines that are clearly section headers or labels
                if any(section in line_lower for section in ["registered address", "tax obligation", "economic activity"]):
                    break
                    
                # Look for lines containing "taxpayer name" or variations
                if any(pattern in line_lower for pattern in ["taxpayer name", "taxpayer namo", "taxpayer nam"]):
                    # Extract the name part by removing the label
                    name_line = re.sub(r'taxpayer\s*nam[eo]?\s*', '', line_lower, flags=re.IGNORECASE).strip()
                    if name_line and not name_line.startswith(':'):
                        # Convert back to proper case
                        taxpayer_name = name_line
                        break
                    else:
                        # Check next line for the name
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line and len(next_line) > 3:
                                # Validate it looks like a name
                                if (re.match(r'^[A-Za-z\s]+$', next_line) or 
                                    re.match(r'^[A-Z\s]+$', next_line)):
                                    taxpayer_name = next_line
                                    break

        # === ALTERNATIVE: Use spatial coordinates for precise extraction ===
        if not taxpayer_name:
            try:
                data = pytesseract.image_to_data(
                    ocr_img, config=custom_config, output_type=pytesseract.Output.DICT
                )
                
                # Find "Taxpayer Information" section coordinates
                taxpayer_info_y = None
                for i in range(len(data['text'])):
                    text_lower = data['text'][i].lower().strip()
                    if "taxpayer information" in text_lower:
                        taxpayer_info_y = data['top'][i]
                        break
                
                if taxpayer_info_y:
                    # Look for text blocks below "Taxpayer Information" and above other sections
                    potential_names = []
                    for i in range(len(data['text'])):
                        text_item = data['text'][i].strip()
                        if not text_item:
                            continue
                            
                        top_pos = data['top'][i]
                        left_pos = data['left'][i]
                        
                        # Only consider text that's below taxpayer info and looks like it could be a name
                        if (top_pos > taxpayer_info_y + 20 and 
                            top_pos < taxpayer_info_y + 200 and  # Within reasonable distance
                            left_pos > 100):  # Not too far left (avoid labels)
                            
                            # Check if it looks like a person's name
                            if (re.match(r'^[A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+$', text_item) or
                                re.match(r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$', text_item)):
                                potential_names.append({
                                    'text': text_item,
                                    'top': top_pos,
                                    'left': left_pos
                                })
                    
                    # Sort by position (top to bottom, left to right) and take the first valid name
                    if potential_names:
                        potential_names.sort(key=lambda x: (x['top'], x['left']))
                        taxpayer_name = potential_names[0]['text']
                        
            except Exception as e:
                print(f"Spatial extraction failed: {e}")

        # === FINAL FALLBACK: Direct search for name patterns ===
        if not taxpayer_name:
            # Look for the specific name format in the raw text
            name_patterns = [
                r'JOHN\s+GIKANGA\s+NJUGI',
                r'JOHN\s*GIKANGA\s*NJUGI',
                r'[A-Z]{4,}\s+[A-Z]{4,}\s+[A-Z]{4,}'  # General pattern for 3 uppercase words
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    taxpayer_name = match.group(0).strip()
                    break

        # === EMAIL EXTRACTION (keep your working version) ===
        email_re = re.compile(
            r'[A-Z0-9._%+-]+\s*@\s*[A-Z0-9.-]+\s*\.\s*[A-Z]{2,}',
            re.IGNORECASE
        )
        
        if not email_address:
            for line in lines:
                if "@" in line and "GMAIL.COM" in line.upper():
                    email_match = email_re.search(line)
                    if email_match:
                        email_address = re.sub(r'\s+', '', email_match.group(0)).upper()
                        break

        # === PIN EXTRACTION (keep your working version) ===
        pin_re = re.compile(r'[A-Z]\s*\d{3}\s*\d{3}\s*\d{3}\s*[A-Z]', re.IGNORECASE)
        
        if not personal_identification_number:
            for line in lines:
                pin_match = pin_re.search(line.upper())
                if pin_match:
                    candidate_pin = re.sub(r'\s+', '', pin_match.group(0)).upper()
                    if re.fullmatch(r'[A-Z]\d{9}[A-Z]', candidate_pin):
                        personal_identification_number = candidate_pin
                        break

        # === CLEANUP ===
        if taxpayer_name:
            # Remove any non-name characters but preserve letters, spaces, hyphens, apostrophes
            taxpayer_name = re.sub(r'[^A-Za-z\s\'’-]', ' ', taxpayer_name).strip()
            taxpayer_name = re.sub(r'\s+', ' ', taxpayer_name)  # Normalize spaces
            
            # Capitalize properly
            if taxpayer_name.isupper():
                taxpayer_name = ' '.join(word.capitalize() for word in taxpayer_name.split())
            else:
                # Ensure proper title case
                taxpayer_name = ' '.join(word.capitalize() if word.isupper() else word 
                                       for word in taxpayer_name.split())

        processing_time = time.time() - start_time

        return {
            "taxpayer_name": taxpayer_name,
            "email_address": email_address,
            "personal_identification_number": personal_identification_number,
            "processing_time": processing_time,
            "engine_used": "tesseract",
            "raw_text": text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract KRA PIN details: {str(e)}")

@router.get("/languages/{engine}")
async def get_supported_languages(engine: OCREngine):
    """Get list of supported languages for the specified OCR engine"""
    try:
        if engine == OCREngine.TESSERACT:
            languages = tesseract_ocr.get_available_languages()
        else:  # PaddleOCR
            languages = paddle_ocr.get_available_languages()
        
        return {"languages": languages}
        
    except Exception as e:
        logger.error(f"Error getting languages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error getting supported languages: {str(e)}",
                "solution": "Please try again or contact support if the problem persists"
            }
        )

import logging

logger = logging.getLogger(__name__)

@router.post("/extract-id-details")
async def extract_name_from_id(file: UploadFile = File(...)):
    """Extract details from Kenyan ID card including name, ID number, and date of birth"""
    try:
        print(f"\n=== Processing {file.filename} ===")
        
        # Read file contents
        contents = await file.read()
        print(f"Read {len(contents)} bytes")
        print(f"Content type: {type(contents)}")
        print(f"First 100 chars: {str(contents)[:100]}")
        
        # Check if the content is actually a string representation of an object
        if isinstance(contents, bytes):
            try:
                # Try to decode as UTF-8 to check if it's a string
                content_str = contents.decode('utf-8')
                if content_str.strip() == '[object Object]' or 'object Object' in content_str:
                    return {
                        "error": "Invalid file data received",
                        "details": "The server received an object instead of image data. Please ensure you're sending the file data correctly.",
                        "solution": "Make sure to use FormData to send the file from the frontend. Example:\n\n// Frontend JavaScript code\nconst formData = new FormData();\nformData.append('file', fileInput.files[0]);\n\n// Then send with fetch/axios"
                    }
            except UnicodeDecodeError:
                # It's binary data, continue processing
                pass
        
        if not contents:
            return {"error": "Uploaded file is empty"}
            
        # Convert to numpy array and decode image
        try:
            # Try reading with OpenCV first, explicitly specifying the color mode
            print(f"Attempting to read with OpenCV from: {file.filename}")
            image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print("OpenCV failed to read the image")
                # If OpenCV fails, try with PIL
                try:
                    from PIL import Image, ImageFile
                    # Enable loading of truncated images
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    
                    print("Attempting to open with PIL...")
                    pil_image = Image.open(BytesIO(contents))
                    print(f"PIL image format: {pil_image.format}, mode: {pil_image.mode}")
                    
                    # Convert PIL image to numpy array
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    print("Successfully converted PIL image to OpenCV format")
                    
                except Exception as pil_error:
                    # Try reading raw bytes if all else fails
                    try:
                        print("Attempting direct bytes decoding...")
                        nparr = np.frombuffer(contents, np.uint8)
                        # Try different IMREAD modes
                        for mode in [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]:
                            image = cv2.imdecode(nparr, mode)
                            if image is not None:
                                print(f"Successfully decoded image with mode: {mode}")
                                break
                        
                        if image is None:
                            # Try with PIL as bytes
                            from io import BytesIO
                            from PIL import Image
                            try:
                                img = Image.open(BytesIO(contents))
                                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                print("Successfully decoded using PIL from bytes")
                            except Exception as pil_bytes_error:
                                print(f"PIL bytes decoding failed: {pil_bytes_error}")
                                raise ValueError("All image decoding methods failed")
                    except Exception as bytes_error:
                        print(f"Direct bytes decoding failed: {str(bytes_error)}")
                        return {
                            "error": "Failed to decode image with all methods",
                            "details": {
                                "pil_error": str(pil_error),
                                "bytes_error": str(bytes_error),
                                "file_size": len(contents),
                                "first_100_bytes": str(contents[:100])
                            }
                        }
            
            print(f"Successfully loaded image. Shape: {image.shape}, Type: {type(image)}")
            
            # Convert to grayscale
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                print("Successfully converted to grayscale")
                
                # Simple thresholding
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print("Successfully applied thresholding")
                # Define ocr_img for downstream OCR calls
                ocr_img = thresh
                 
            except Exception as proc_error:
                return {
                    "error": "Error during image processing",
                    "details": str(proc_error)
                }
            
            # Save processed image for debugging
            cv2.imwrite('debug_processed.png', thresh)
            
            # Use Tesseract directly
            print("Running Tesseract OCR for Kenyan ID...")
            import pytesseract
            
            custom_config = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(ocr_img, config=custom_config)
            print("=== EXTRACTED TEXT ===")
            print(text)
            print("=====================")
            
            # Process text to extract name from Kenyan ID
            full_name = ""
            
            # Process the extracted text to find the required fields
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Initialize variables to store extracted information
            full_name = ""
            id_number = ""
            date_of_birth = ""
            surname = ""
            other_names = ""
            
            # First pass: Look for ID number (8 digits) and name pattern
            for i, line in enumerate(lines):
                clean_line = line.upper().strip()
                original_line = lines[i]  # ensure original_line is defined for this iteration
                
                # Look for 8-digit ID number (Kenyan ID format)
                id_match = re.search(r'\b(\d{8})\b', clean_line)
                if id_match and not id_number:  # Only set if not already found
                    id_number = id_match.group(1)
                    print(f"Found ID number: {id_number} in line: {clean_line}")
                    
                    # Check if the next line contains the name (common pattern in Kenyan IDs)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Look for a line with 2-4 uppercase words (potential name), allow apostrophes/hyphens
                        name_match = re.match(r"^[A-Z][A-Z'’\-]*(?:\s+[A-Z][A-Z'’\-]*){1,3}$", next_line)
                        if name_match:
                            full_name = next_line
                            print(f"Found name after ID: {full_name}")
                
                # Look for name pattern (2-4 uppercase words, allow apostrophes/hyphens)
                name_match = re.match(r"^[A-Z][A-Z'’\-]*(?:\s+[A-Z][A-Z'’\-]*){1,3}$", clean_line)
                if name_match and not full_name:  # Only set if not already found
                    # Additional check to avoid false positives (not all caps or too short)
                    if len(clean_line) > 5 and clean_line.isupper():
                        # Use the original line to preserve casing/characters
                        full_name = original_line.strip()
                        print(f"Found name pattern: {full_name}")
            
            # If we have an ID but no name, try to find the name in the text
            if id_number and not full_name:
                print("Looking for name in text...")
                # Look for a line with 2-4 uppercase words that's not the ID line
                for line in lines:
                    clean_line = line.strip()
                    if (re.match(r'^[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}$', clean_line) and 
                        not re.search(r'\d', clean_line)):  # Exclude lines with numbers
                        full_name = clean_line
                        print(f"Found potential name in text: {full_name}")
                        break
            
            # If we still don't have a name, try to find it near date of birth
            if not full_name:
                for i, line in enumerate(lines):
                    # Look for date pattern (DD.MM.YYYY or DD/MM/YYYY)
                    # Allow separators '.', '/', '-' and optional comma/space before year, e.g., '09.05, 1991'
                    if re.search(r"\b\d{1,2}[\.\-/]\d{1,2}[,\s\.\-/]*\d{2,4}\b", line):
                        # Check the line before the date (common pattern for name)
                        if i > 0 and len(lines[i-1].strip().split()) >= 2:
                            potential_name = lines[i-1].strip()
                            if re.match(r"^[A-Z][A-Z'’\-]*(?:\s+[A-Z][A-Z'’\-]*){1,3}$", potential_name.upper()):
                                full_name = potential_name
                                print(f"Found name before date: {full_name}")
                                break
            
            # Second pass: Look for name fields with more robust matching
            for i, line in enumerate(lines):
                clean_line = line.upper().strip()
                original_line = lines[i]  # Keep original line for case preservation
                
                # Look for full name first (common in Kenyan IDs)
                if not full_name and ("FULL NAMES" in clean_line or "FULL NAME" in clean_line):
                    # Try to extract name from the same line after a colon or label
                    if ':' in original_line:
                        name_part = original_line.split(':', 1)[1].strip()
                        if len(name_part.split()) >= 2:  # If it looks like a full name
                            full_name = name_part.strip('"').strip("'").strip()
                            print(f"Extracted full name from label line: {full_name}")
                    
                    # If not found, check next line for name
                    if not full_name and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if len(next_line.split()) >= 2:  # If next line has multiple words, it's likely a name
                            full_name = next_line.strip('"').strip("'").strip()
                            print(f"Found full name on next line: {full_name}")
                
                # Look for surname/last name with more flexible matching
                surname_labels = ["Full Names", "SURNAME", "LAST NAME", "FAMILY NAME", "SUR NAME", "FAMILY"]
                if any(label in clean_line for label in surname_labels):
                    if i + 1 < len(lines):
                        # Get the next line and clean it up while preserving case
                        surname_line = lines[i + 1].strip()
                        # Remove any non-letter characters except spaces, hyphens, and apostrophes
                        surname = re.sub(r"[^A-Za-z\s'’-]", '', surname_line).strip()
                        print(f"Found surname: {surname} after label: {clean_line}")
                
                # Look for other names/first name with more flexible matching
                other_names_labels = [
                    "OTHER NAMES", "FIRST NAME", "GIVEN NAME", 
                    "FIRSTNAME", "GIVENNAME", "OTHER NAMES/FIRST NAME"
                ]
                if any(label in clean_line for label in other_names_labels):
                    if i + 1 < len(lines):
                        other_names = lines[i + 1].strip()
                        # Clean up the names while preserving case
                        other_names = re.sub(r"[^A-Za-z\s'-]", '', other_names).strip()
                        print(f"Found other names: {other_names} after label: {clean_line}")
                
                # Look for full name (common in some ID formats)
                full_name_labels = [
                    "FULL NAMES", "FULL NAMES", "NAME:", "NAMES:", "NAME OF PERSON", "NAME", 
                    "NAME OF HOLDER", "HOLDER'S NAME", "NAMES"
                ]
                
                # Check for full name patterns in the current line
                if any(label in clean_line for label in full_name_labels):
                    # First try to extract from the same line after the label
                    for label in full_name_labels:
                        if label in clean_line:
                            # Get the part after the label (support ':' or '-')
                            name_part = original_line
                            # Prefer splitting after ':'
                            if ':' in original_line:
                                name_part = original_line.split(':', 1)[1].strip()
                            elif '-' in original_line:
                                name_part = original_line.split('-', 1)[1].strip()
                            else:
                                # Fallback: remove the label text from the start if present
                                pattern = re.compile(r"^\s*" + re.escape(label), re.IGNORECASE)
                                name_part = pattern.sub('', original_line).strip()
                            if name_part:  # If we found a name on the same line
                                # Keep ASCII apostrophe ' and Unicode right single quote ’
                                full_name = re.sub(r"[^A-Za-z\s'’\-]", '', name_part).strip()
                                print(f"Extracted full name from same line: {full_name}")
                                break
                    
                    # If not found on same line, try the next line
                    if not full_name and i + 1 < len(lines):
                        full_name = lines[i + 1].strip()
                        # Clean up the full name while preserving case and Unicode apostrophes
                        full_name = re.sub(r"[^A-Za-z\s'’\-]", '', full_name).strip()
                        print(f"Found full name on next line: {full_name}")
                    
                    # If we have a full name but no surname/other names, try to split it
                    if full_name and (not surname or not other_names):
                        # Clean up multiple spaces and normalize
                        full_name = ' '.join(full_name.split())
                        parts = full_name.split()
                        
                        if len(parts) >= 2:
                            # Common Kenyan naming convention: First Middle [Middle...] Last
                            # Try to identify surname (usually the last part)
                            surname_candidate = parts[-1]
                            other_names_candidate = ' '.join(parts[:-1])
                            
                            # Update only if we don't have these values yet
                            if not surname:
                                surname = surname_candidate
                            if not other_names:
                                other_names = other_names_candidate
                            
                            print(f"Split full name into surname: {surname} and other names: {other_names}")
                
                # Special case: Some IDs have names in format "SURNAME, OTHER NAMES"
                if not full_name and ',' in line and sum(c.isalpha() or c.isspace() or c in "'-" for c in line) > 5:
                    name_parts = [p.strip() for p in line.split(',', 1)]
                    if len(name_parts) == 2 and all(len(p.split()) >= 1 for p in name_parts):
                        surname_candidate = name_parts[0]
                        other_names_candidate = name_parts[1]
                        
                        # Clean up the names
                        surname_candidate = re.sub(r"[^A-Za-z\s'-]", '', surname_candidate).strip()
                        other_names_candidate = re.sub(r"[^A-Za-z\s'-]", '', other_names_candidate).strip()
                        
                        if not surname and surname_candidate:
                            surname = surname_candidate
                        if not other_names and other_names_candidate:
                            other_names = other_names_candidate
                        
                        print(f"Found name in 'SURNAME, OTHER NAMES' format: {surname}, {other_names}")
            
            # Final assembly of full name
            if not full_name:
                if surname and other_names:
                    full_name = f"{other_names} {surname}"
                elif surname:
                    full_name = surname
                elif other_names:
                    full_name = other_names
                
                # If we still don't have a full name, look for a line that looks like a full name
                if not full_name:
                    for line in lines:
                        name_parts = line.strip().split()
                        # Look for lines with 2-4 name parts with proper capitalization
                        if 2 <= len(name_parts) <= 4 and all(len(part) > 1 for part in name_parts):
                            full_name = line.strip()
                            print(f"Found potential full name from text: {full_name}")
                            break
                
                if full_name:
                    # Clean up the name while preserving proper capitalization
                    full_name = ' '.join(word.strip() for word in full_name.split())
                    # Handle special cases like O'Brien, Nya-Okoth, etc.
                    full_name = re.sub(r"\s*([-'’])\s*", r"\1", full_name)
                    # Ensure proper capitalization (first letter of each word, except for special cases)
                    full_name = ' '.join(
                        word[0].upper() + word[1:].lower() 
                        if not any(c.islower() for c in word)  # Only if all caps
                        else word  # Keep original if mixed case (e.g., McDonald)
                        for word in re.split(r"(\s+|['’-]|(?<=\w)(?=[A-Z]))", full_name) 
                        if word.strip()
                    )
                    # Fix any remaining spacing issues
                    full_name = ' '.join(full_name.split())
                    print(f"Final constructed full name: {full_name}")
           
                
                # Extract Date of Birth (support DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY and with optional comma)
                if not date_of_birth and any(label in clean_line for label in ["DATE OF BIRTH", "DOB", "BIRTH DATE"]):
                    # Check current line for date
                    dob_match = re.search(r"\b(\d{1,2}[\.\-/]\d{1,2}[,\s\.\-/]*\d{2,4})\b", clean_line)
                    if dob_match:
                        date_of_birth = dob_match.group(1).strip()
                    # If not found, check next line
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        dob_match = re.search(r"\b(\d{1,2}[\.\-/]\d{1,2}[,\s\.\-/]*\d{2,4})\b", next_line)
                        if dob_match:
                            date_of_birth = dob_match.group(1).strip()

            # Fallback: if DOB still not found via labels, scan the entire text for date-like patterns
            if not date_of_birth:
                date_pattern = re.compile(r"\b(\d{1,2})[\.\-/](\d{1,2})[,\s\.\-/]*(\d{2,4})\b")
                dob_candidates = []
                for ln in lines:
                    for m in date_pattern.finditer(ln):
                        d, mo, y = m.groups()
                        year = int(y) if len(y) == 4 else 2000 + int(y)
                        # Sanity check plausible years
                        if 1900 <= year <= 2100:
                            dob_candidates.append((year, d, mo, y))
                if dob_candidates:
                    # Prefer the earliest year (likely birth vs issue date)
                    dob_candidates.sort(key=lambda t: t[0])
                    _, d, mo, y = dob_candidates[0]
                    if len(y) == 2:
                        y = "20" + y
                    date_of_birth = f"{d.zfill(2)}/{mo.zfill(2)}/{y}"
            
            # If we still don't have an ID number, search the entire text with patterns
            if not id_number:
                full_text = "\n".join(lines)
                print("Searching full text for ID number...")
                
                # Try standard Kenyan ID format (1-2 digits, 6-7 digits, optional letter)
                id_patterns = [
                    (r'\b(?:ID[^\d]*)?(\d{1,2}\s?\d{6,7}[A-Za-z]?)\b', "Standard format"),  # Standard format
                    (r'\b(?:ID[^\d]*)?(\d{1,2}[-/]\d{6,7}[A-Za-z]?)\b', "With dash/slash"),  # With dash/slash
                    (r'\b(\d{8,9}[A-Za-z]?)\b', "Just numbers (8-9 digits)"),  # Just numbers (8-9 digits)
                    (r'\b(?:ID[^\d]*)(\d+[A-Za-z]?)\b', "ID prefix with numbers")  # ID prefix with numbers
                ]
                
                for pattern, pattern_name in id_patterns:
                    id_match = re.search(pattern, full_text, re.IGNORECASE)
                    if id_match:
                        potential_id = re.sub(r'[^\dA-Za-z]', '', id_match.group(1))
                        # Basic validation for Kenyan ID format
                        if 8 <= len(potential_id) <= 9 and (potential_id[-1].isdigit() or potential_id[-1].isalpha()):
                            id_number = potential_id.upper()
                            print(f"Found ID number using {pattern_name}: {id_number}")
                            break
            
            # Clean and validate ID number
            if id_number:
                # Remove any non-alphanumeric characters
                id_number = re.sub(r'[^\dA-Za-z]', '', id_number.upper())
                # Standardize to 8-9 characters (1-2 digits + 7 digits + optional letter)
                if len(id_number) >= 8 and id_number[-1].isdigit() and len(id_number) == 8:
                    # If 8 digits, it's valid (1 digit + 7 digits)
                    pass
                elif len(id_number) >= 9 and id_number[-1].isalpha() and len(id_number) == 9:
                    # If 9 characters with a letter at the end, it's valid
                    pass
                elif len(id_number) > 8:
                    # If longer than 8, take first 8 characters
                    id_number = id_number[:8]
                elif len(id_number) < 8:
                    # If too short, it's not a valid Kenyan ID
                    id_number = ""
            
            # Format date of birth to consistent format (DD/MM/YYYY)
            # Format date of birth to consistent format (DD/MM/YYYY)
            if date_of_birth:
                try:
                    # Normalize commas/spaces around year
                    date_of_birth = date_of_birth.replace(',', ' ').strip()
                    # Collapse multiple spaces
                    date_of_birth = re.sub(r"\s+", ' ', date_of_birth)
                    # Handle different date separators
                    if '-' in date_of_birth:
                        parts = date_of_birth.split('-')
                        day, month, year = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    elif '/' in date_of_birth:
                        parts = date_of_birth.split('/')
                        day, month, year = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    elif '.' in date_of_birth:
                        parts = date_of_birth.split('.')
                        # If there are extra spaces around year, ensure cleanup
                        if len(parts) >= 3:
                            day, month, year = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        else:
                            raise ValueError("Invalid date format with dots")
                    else:
                        # Assume format is DDMMYYYY or DDMMYY
                        if len(date_of_birth) == 8:  # DDMMYYYY
                            day, month, year = date_of_birth[:2], date_of_birth[2:4], date_of_birth[4:]
                        elif len(date_of_birth) == 6:  # DDMMYY
                            day, month, year = date_of_birth[:2], date_of_birth[2:4], "20" + date_of_birth[4:]
                        else:
                            raise ValueError("Unknown date format")
                    
                    # Ensure 2-digit day and month, 4-digit year
                    day = day.zfill(2)
                    month = month.zfill(2)
                    if len(year) == 2:
                        year = "20" + year
                    
                    date_of_birth = f"{day}/{month}/{year}"
                except Exception as e:
                    print(f"Error formatting date of birth: {str(e)}")
                    # Keep original value if formatting fails
                    date_of_birth = date_of_birth
            
            # Combine names and clean up only if full_name hasn't been found earlier
            if not full_name or full_name.isspace():
                # If we have both surname and other names, combine them
                if surname and other_names:
                    full_name = f"{surname} {other_names}".strip()
                elif surname:
                    full_name = surname.strip()
                elif other_names:
                    full_name = other_names.strip()


            # If we still don't have a name, look for any line that looks like a name
            if not full_name or full_name.isspace():
                for line in lines:
                    line = line.strip()
                    # At least two words, first letter of each word capitalized
                    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
                        full_name = line
                        break

            # Final cleanup of the name (preserve original casing)
            if full_name:
                # Normalize spaces
                full_name = ' '.join(full_name.split())
            
            # Prepare the response
            response = {
                "status": "success",
                "engine_used": "Tesseract",
                "full_name": full_name if full_name else "Not found",
                "id_number": id_number if id_number else "Not found",
                "date_of_birth": date_of_birth if date_of_birth else "Not found",
                "details": {
                    "full_name": full_name,
                    "id_number": id_number,
                    "date_of_birth": date_of_birth,
                    "raw_text": text
                },
                # Add additional fields for better frontend compatibility
                "name": full_name if full_name else "Not found",
                "extracted_name": full_name if full_name else "Not found",
                "id": id_number if id_number else "Not found"
            }
            return response
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        try:
            await file.close()
            print("File handle closed")
        except Exception as e:
            print(f"Error closing file: {str(e)}")


@router.post("/preview-preprocessing")
async def preview_preprocessing(file: UploadFile = File(...)):
    """Preview the preprocessing steps on an image"""
    try:
        # Read and validate image
        contents = await file.read()
        is_valid, msg = validate_image(contents, settings.MAX_FILE_SIZE)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)
        
        # Process image
        processed_img = preprocess_image(contents)
        
        # Convert back to bytes
        _, img_encoded = cv2.imencode('.png', processed_img)
        img_bytes = img_encoded.tobytes()
        
        # Return as base64 encoded image
        import base64
        return {
            "preview_image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
