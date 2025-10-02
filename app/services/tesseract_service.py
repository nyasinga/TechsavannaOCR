import os
import pytesseract
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import time
import subprocess
import sys

from fastapi import HTTPException
from ..core.config import settings

class TesseractOCR:
    def __init__(self):
        # Try to find Tesseract in common locations if not in PATH
        self.tesseract_path = None
        
        # Check if Tesseract is in PATH
        try:
            if sys.platform == 'win32':
                # Common Windows installation paths
                common_paths = [
                    os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Tesseract-OCR', 'tesseract.exe'),
                    os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Tesseract-OCR', 'tesseract.exe'),
                    'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
                    'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
                    'tesseract'  # Try if it's in PATH
                ]
            else:
                # Common Unix paths
                common_paths = [
                    '/usr/bin/tesseract',
                    '/usr/local/bin/tesseract',
                    '/opt/homebrew/bin/tesseract',
                    'tesseract'  # Try if it's in PATH
                ]
            
            # Check each possible path
            for path in common_paths:
                try:
                    subprocess.run(
                        [path, '--version'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    self.tesseract_path = path
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
            
            # If still not found, try to use the one from settings
            if not self.tesseract_path and hasattr(settings, 'TESSERACT_CMD'):
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
                self.tesseract_path = settings.TESSERACT_CMD
            
            # Final check if Tesseract is accessible
            if not self.tesseract_path:
                raise RuntimeError("Tesseract not found in PATH or common installation directories")
                
        except Exception as e:
            error_msg = (
                "Tesseract OCR is not installed or not found in your PATH. "
            )
            raise RuntimeError(error_msg) from e
    
    def extract_text(
        self, 
        image: np.ndarray, 
        language: str = 'eng',
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image: Numpy array of the image
            language: Language code (e.g., 'eng', 'fra', 'spa')
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        start_time = time.time()
        
        # Preprocess image if requested
        if preprocess:
            from ..utils.image_processing import preprocess_image
            processed_img = preprocess_image(
                cv2.imencode('.png', image)[1].tobytes()
            )
        else:
            processed_img = image
        
        # Extract text with confidence scores
        data = pytesseract.image_to_data(
            processed_img,
            output_type=pytesseract.Output.DICT,
            lang=language,
            config='--psm 6'  # Assume a single uniform block of text
        )
        
        # Process the results
        text_blocks = []
        word_boxes = []
        
        for i, word in enumerate(data['text']):
            if int(data['conf'][i]) > 0:  # Only include words with confidence > 0
                text_blocks.append(word)
                
                # Get word bounding box and confidence
                x, y, w, h = (
                    int(data['left'][i]),
                    int(data['top'][i]),
                    int(data['width'][i]),
                    int(data['height'][i])
                )
                
                word_boxes.append({
                    'text': word,
                    'confidence': float(data['conf'][i]) / 100.0,  # Convert to 0-1 range
                    'position': [x, y, x + w, y + h]  # [x1, y1, x2, y2]
                })
        
        # Calculate average confidence
        confidences = [box['confidence'] for box in word_boxes]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': ' '.join(text_blocks).strip(),
            'confidence': avg_confidence,
            'words': word_boxes,
            'processing_time': time.time() - start_time
        }
    
    def get_available_languages(self) -> List[str]:
        """Get list of available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return langs
        except:
            # Fallback to common languages if language list can't be retrieved
            return ['eng', 'fra', 'spa', 'deu', 'ita', 'por', 'rus']

# Singleton instance
tesseract_ocr = TesseractOCR()
