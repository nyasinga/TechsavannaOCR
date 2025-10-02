import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageEnhance
import io

def validate_image(file: bytes, max_size: int) -> Tuple[bool, str]:
    """Validate image file size and type"""
    if len(file) > max_size:
        return False, f"File size exceeds maximum limit of {max_size} bytes"
    
    # Check file signature
    if file.startswith(b'\xff\xd8'):
        return True, "image/jpeg"
    elif file.startswith(b'\x89PNG\r\n\x1a\n'):
        return True, "image/png"
    elif file.startswith(b'BM'):
        return True, "image/bmp"
    elif file.startswith(b'II*\x00') or file.startswith(b'MM\x00*'):
        return True, "image/tiff"
    
    return False, "Unsupported file format"

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for better OCR results
    
    Args:
        image_data: Binary image data
        
    Returns:
        Preprocessed numpy array image
        
    Raises:
        ValueError: If image processing fails at any step
    """
    try:
        if not image_data or len(image_data) == 0:
            raise ValueError("Empty image data provided")
            
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        if len(nparr) == 0:
            raise ValueError("Failed to convert image data to numpy array")
            
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data")
            
        print(f"Image shape before preprocessing: {img.shape}")
        
        # Convert to grayscale
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            raise ValueError(f"Failed to convert image to grayscale: {str(e)}")
        
        # Denoising
        try:
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        except cv2.error as e:
            print(f"Warning: Denoising failed, continuing without it. Error: {str(e)}")
            denoised = gray  # Continue with non-denoised image
        
        # Adaptive thresholding
        try:
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        except cv2.error as e:
            raise ValueError(f"Adaptive thresholding failed: {str(e)}")
        
        # Apply dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply deskewing
        try:
            deskewed = deskew(opening)
            print(f"Image preprocessing completed successfully. Output shape: {deskewed.shape if deskewed is not None else 'None'}")
            return deskewed
            
        except Exception as e:
            print(f"Warning: Deskewing failed, returning preprocessed image without deskewing. Error: {str(e)}")
            return opening
            
    except Exception as e:
        error_msg = f"Error in image preprocessing: {str(e)}"
        print(error_msg)
        # Try to return the original image if preprocessing fails
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) or np.array([])
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Deskew an image using its skew angle
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Deskewed image as numpy array
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    # Compute minimum area rectangle that contains all the non-zero pixels
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle for proper rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def bytes_to_pil_image(image_data: bytes) -> Image.Image:
    """Convert bytes to PIL Image"""
    return Image.open(io.BytesIO(image_data))

def pil_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """Convert PIL Image to bytes"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()
