import re
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from app.schemas.classifier import DocumentType, ClassScore

class DocumentClassifier:
    """
    Classifies documents as either KRA PIN certificates or National IDs
    based on text patterns and features.
    """
    
    # Keywords that strongly indicate a KRA PIN certificate
    KRA_KEYWORDS = [
        'kenya revenue authority', 'kra', 'pin certificate', 'personal identification number',
        'taxpayer', 'certificate of registration', 'tax obligations', 'vat', 'paye'
    ]
    
    # Keywords that strongly indicate a National ID
    ID_KEYWORDS = [
        'republic of kenya', 'national id', 'identity card', 'id no', 'date of birth',
        'date of issue', 'county', 'district', 'gender', 'nationality'
    ]
    
    # Common patterns for KRA PIN
    KRA_PATTERNS = [
        r'[A-Z]\d{6}[A-Z]',  # KRA PIN format
        r'certificate\s*no[.:]?\s*[A-Z0-9]+',
        r'tax\s*obligations?',
    ]
    
    # Common patterns for National ID
    ID_PATTERNS = [
        r'id\s*no[.:]?\s*[0-9]{8,9}',  # Kenyan ID number format
        r'date of (birth|issue)[.:]',
        r'place of (birth|issue)[.:]',
    ]
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the document classifier.
        
        Args:
            confidence_threshold: Minimum confidence score (0-1) for a prediction to be considered valid
        """
        self.confidence_threshold = confidence_threshold
        self.kra_keyword_patterns = [re.compile(pat, re.IGNORECASE) for pat in self.KRA_PATTERNS]
        self.id_keyword_patterns = [re.compile(pat, re.IGNORECASE) for pat in self.ID_PATTERNS]
    
    def _calculate_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each document type based on text patterns.
        
        Args:
            text: The text content to analyze
            
        Returns:
            Dictionary with document type scores
        """
        text = text.lower()
        scores = {
            DocumentType.KRA_PIN_CERTIFICATE: 0.0,
            DocumentType.NATIONAL_ID: 0.0,
            DocumentType.UNKNOWN: 0.0
        }
        
        # Check for KRA patterns
        kra_matches = 0
        for pattern in self.kra_keyword_patterns:
            if pattern.search(text):
                kra_matches += 1
        
        # Check for ID patterns
        id_matches = 0
        for pattern in self.id_keyword_patterns:
            if pattern.search(text):
                id_matches += 1
        
        # Simple keyword matching
        kra_keywords_found = sum(1 for kw in self.KRA_KEYWORDS if kw in text)
        id_keywords_found = sum(1 for kw in self.ID_KEYWORDS if kw in text)
        
        # Calculate scores (weighted sum of pattern matches and keyword matches)
        kra_score = (kra_matches * 0.6) + (kra_keywords_found * 0.4)
        id_score = (id_matches * 0.6) + (id_keywords_found * 0.4)
        
        # Normalize scores
        total = kra_score + id_score + 0.1  # Add small value to avoid division by zero
        
        scores[DocumentType.KRA_PIN_CERTIFICATE] = kra_score / total
        scores[DocumentType.NATIONAL_ID] = id_score / total
        scores[DocumentType.UNKNOWN] = 1.0 - max(kra_score, id_score) / total
        
        return scores
    
    def _get_top_predictions(self, scores: Dict[str, float], top_k: int = 3) -> List[ClassScore]:
        """
        Get top-k predictions from the scores.
        
        Args:
            scores: Dictionary of document type scores
            top_k: Number of top predictions to return
            
        Returns:
            List of ClassScore objects sorted by score in descending order
        """
        sorted_scores = sorted(
            [ClassScore(label=doc_type, score=score) 
             for doc_type, score in scores.items()],
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_scores[:top_k]
    
    def classify_text(self, text: str, top_k: int = 3) -> Tuple[DocumentType, float, List[ClassScore]]:
        """
        Classify a document based on its text content.
        
        Args:
            text: The text content to classify
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (predicted_type, confidence, predictions)
        """
        start_time = time.time()
        
        # Calculate scores
        scores = self._calculate_scores(text)
        
        # Get top predictions
        predictions = self._get_top_predictions(scores, top_k)
        
        # Get the prediction with highest confidence
        predicted_type = DocumentType(predictions[0].label)
        confidence = predictions[0].score
        
        # Check if we need human review
        needs_review = confidence < self.confidence_threshold
        
        processing_time = time.time() - start_time
        
        return {
            "document_type": predicted_type,
            "confidence": confidence,
            "predictions": predictions,
            "needs_review": needs_review,
            "processing_time": processing_time,
            "metadata": {
                "version": "1.0.0",
                "model": "document-classifier-v1"
            }
        }

# Singleton instance
classifier = DocumentClassifier()
