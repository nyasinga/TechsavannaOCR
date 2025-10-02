from typing import Dict, Any, List, Optional
import time
import numpy as np

class PaddleOCRService:
    def __init__(self):
        self.ocr = None
        self.initialized = False
    
    def initialize(self):
        """Lazy initialization of PaddleOCR to avoid slow startup"""
        if not self.initialized:
            from paddleocr import PaddleOCR
            from ..core.config import settings
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Default language, can be overridden
                use_gpu=settings.USE_GPU,
                show_log=False
            )
            self.initialized = True
    
    def extract_text(
        self, 
        image: np.ndarray, 
        language: str = 'en',
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from image using PaddleOCR
        
        Args:
            image: Numpy array of the image
            language: Language code (e.g., 'en', 'fr', 'es')
            preprocess: Whether to preprocess the image (handled by PaddleOCR internally)
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        self.initialize()
        start_time = time.time()
        
        try:
            # Convert image to RGB if needed (PaddleOCR expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:  # Already in RGB
                rgb_image = image
            elif len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:  # BGR or other
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            result = self.ocr.ocr(rgb_image, cls=True)
            
            # Process results
            text_blocks = []
            word_boxes = []
            total_confidence = 0.0
            valid_boxes = 0
            
            if result is not None:
                for line in result:
                    if line and len(line) > 0:
                        for word_info in line:
                            if word_info and len(word_info) >= 2:
                                # Extract bounding box points and text with confidence
                                points = word_info[0]
                                text, confidence = word_info[1]
                                
                                # Flatten points to [x1, y1, x2, y2, ...]
                                flat_points = [coord for point in points for coord in point]
                                
                                # Calculate bounding box
                                x_coords = [points[i][0] for i in range(4)]
                                y_coords = [points[i][1] for i in range(4)]
                                x1, x2 = min(x_coords), max(x_coords)
                                y1, y2 = min(y_coords), max(y_coords)
                                
                                if text.strip() and confidence > 0:
                                    text_blocks.append(text)
                                    word_boxes.append({
                                        'text': text,
                                        'confidence': float(confidence),
                                        'position': [x1, y1, x2, y2],
                                        'polygon': flat_points
                                    })
                                    total_confidence += confidence
                                    valid_boxes += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_boxes if valid_boxes > 0 else 0.0
            
            return {
                'text': ' '.join(text_blocks).strip(),
                'confidence': avg_confidence,
                'words': word_boxes,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'words': [],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def get_available_languages(self) -> List[str]:
        """Get list of available PaddleOCR languages"""
        # PaddleOCR supports these languages
        return [
            'en', 'ch', 'french', 'german', 'korean', 'japan', 'chinese_cht',
            'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]

# Singleton instance
paddle_ocr = PaddleOCRService()
