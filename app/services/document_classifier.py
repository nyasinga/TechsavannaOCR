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
    
    # Keywords for document classification
    DOCUMENT_KEYWORDS = {
        DocumentType.KRA_PIN_CERTIFICATE: [
            'kenya revenue authority', 'kra', 'pin certificate', 'personal identification number',
            'taxpayer', 'tax obligations', 'vat', 'paye', 'tax compliance'
        ],
        DocumentType.NATIONAL_ID: [
            'republic of kenya', 'national id', 'identity card', 'id no', 'date of birth',
            'date of issue', 'county', 'district', 'gender', 'nationality', 'place of birth'
        ],
        DocumentType.CERTIFICATE_OF_REGISTRATION: [
            'certificate of registration', 'certificate of incorporation', 'company registration',
            'business registration', 'certificate no', 'registered under the companies act'
        ],
        DocumentType.BUSINESS_PERMIT: [
            # Document title/header
            'business permit', 'single business permit', 'annual business permit',
            'trading license', 'business license', 'permit to operate',
            
            # Issuing authority
            'county government', 'county government of', 'municipal council',
            'city council', 'town council', 'department of trade',
            'department of commerce', 'licensing department',
            
            # Document identifiers
            'permit no', 'permit number', 'license no', 'license number',
            'receipt no', 'receipt number', 'reference no', 'reference number',
            'permit reference', 'license reference',
            
            # Business information
            'business name', 'trading as', 'trade name', 'business activity',
            'nature of business', 'type of business', 'business category',
            'business location', 'business address', 'premises',
            
            # Location details
            'ward', 'sub-county', 'subcounty', 'county', 'location', 'sub-location',
            'estate', 'building', 'street', 'road', 'avenue', 'lane', 'plot',
            
            # Dates and validity
            'date of issue', 'date issued', 'issued on', 'date of expiry',
            'expiry date', 'valid until', 'valid from', 'valid to',
            'valid for the period', 'validity period', 'financial year',
            
            # Financial information
            'permit fee', 'license fee', 'total amount', 'amount paid',
            'payment details', 'payment reference', 'receipt',
            'kra pin', 'vat no', 'vat number',
            
            # Owner/Proprietor details
            'owner name', 'proprietor name', 'director', 'partner',
            'proprietor details', 'owner details',
            
            # Official sections and stamps
            'for official use only', 'authorized signatory', 'county seal',
            'official stamp', 'approval', 'endorsement',
            
            # Contact information
            'telephone', 'phone', 'mobile', 'email', 'p.o. box', 'postal address',
            
            # Regulatory information
            'conditions', 'terms and conditions', 'regulations', 'by-laws',
            'act no', 'section', 'subsection',
            
            # Common headers and footers
            'republic of kenya', 'county government', 'county logo',
            'this permit is issued under', 'in accordance with',
            'any person who forges or fraudulently alters',
            
            # Additional common phrases
            'this is to certify that', 'is hereby licensed to',
            'is authorized to operate', 'is permitted to carry on the business of'
        ],
        DocumentType.CR12: [
            'form cr12', 'certificate of registration of members', 'list of members',
            'share capital', 'directors', 'company seal', 'certified by registrar of companies'
        ],
        DocumentType.SERVICE_APPLICATION_FORM: [
            'application form', 'service application', 'form no', 'applicant details',
            'purpose of application', 'declaration', 'official use only'
        ]
    }
    
    # Common patterns for document matching
    DOCUMENT_PATTERNS = {
        DocumentType.KRA_PIN_CERTIFICATE: [
            r'[A-Z]\d{6}[A-Z]',  # KRA PIN format
            r'certificate\s*no[.:]?\s*[A-Z0-9]+',
            r'tax\s*obligations?',
        ],
        DocumentType.NATIONAL_ID: [
            r'id\s*no[.:]?\s*[0-9]{8,9}',  # Kenyan ID number format
            r'date of (birth|issue)[.:]',
            r'place of (birth|issue)[.:]',
        ],
        DocumentType.CERTIFICATE_OF_REGISTRATION: [
            r'certificate\s*of\s*registration\s*no[.:]?\s*[A-Z0-9/]+',
            r'registered\s*under\s*the\s*companies\s*act',
            r'company\s*registration\s*no[.:]?\s*[A-Z0-9/]+'
        ],
        DocumentType.BUSINESS_PERMIT: [
            # Permit number patterns
            r'permit\s*(?:no\.?|number)[\s:]*([A-Z0-9/-]+)',
            r'business\s*permit\s*(?:no\.?|number)[\s:]*([A-Z0-9/-]+)',
            r'(?:single\s*)?business\s*permit[\s:]*([A-Z0-9/-]+)',
            
            # Date patterns
            r'date\s*of\s*issue[\s:]*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})',
            r'date\s*of\s*expir(?:y|ation)[\s:]*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})',
            r'valid\s*(?:until|till|to)[\s:]*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})',
            r'valid\s*for\s*the\s*period[\s:]*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})\s*to\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})',
            
            # Amount patterns
            r'total\s*amount[\s:]*K?SH[\s:]*([\d,]+(?:\.\d{2})?)',
            r'permit\s*fee[\s:]*K?SH[\s:]*([\d,]+(?:\.\d{2})?)',
            r'amount\s*paid[\s:]*K?SH[\s:]*([\d,]+(?:\.\d{2})?)',
            
            # Business details
            r'business\s*name[\s:]*([^\n]+)',
            r'trading\s*as[\s:]*([^\n]+)',
            r'business\s*activity[\s:]*([^\n]+)',
            r'nature\s*of\s*business[\s:]*([^\n]+)',
            r'business\s*address[\s:]*([^\n]+)',
            r'location[\s:]*([^\n]+)',
            r'ward[\s:]*([^\n]+)',
            r'sub-?county[\s:]*([^\n]+)',
            
            # Contact information
            r'phone\s*(?:no\.?|number)[\s:]*([+\d\s-]+)',
            r'email[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            
            # Official sections
            r'county\s*government\s*of[\s:]*([^\n]+)',
            r'department\s*of[\s:]*([^\n]+)',
            r'for\s*official\s*use\s*only',
            
            # Receipt/Payment information
            r'receipt\s*(?:no\.?|number)[\s:]*([A-Z0-9/-]+)',
            r'payment\s*reference[\s:]*([A-Z0-9/-]+)',
            r'transaction\s*id[\s:]*([A-Z0-9-]+)'
        ],
        DocumentType.CR12: [
            r'form\s*cr\s*12',
            r'certified\s*extract\s*from\s*the\s*register',
            r'certificate\s*of\s*registration\s*of\s*members',
            r'share\s*capital[\s:]+[\d,]+'
        ],
        DocumentType.SERVICE_APPLICATION_FORM: [
            r'application\s*form',
            r'service\s*request',
            r'application\s*no[.:]?\s*[A-Z0-9/-]+',
            r'date[\s:]+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}'
        ]
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the document classifier.
        
        Args:
            confidence_threshold: Minimum confidence score (0-1) for a prediction to be considered valid
        """
        self.confidence_threshold = confidence_threshold
        # Compile all regex patterns for faster matching
        self._compiled_patterns = {
            doc_type: [re.compile(pat, re.IGNORECASE) for pat in patterns]
            for doc_type, patterns in self.DOCUMENT_PATTERNS.items()
        }
    
    def _calculate_scores(self, text: str) -> Dict[DocumentType, float]:
        """
        Calculate confidence scores for each document type based on text patterns.
        
        Args:
            text: The text content to analyze (case-insensitive)
            
        Returns:
            Dictionary mapping document types to their confidence scores (0-1)
        """
        text = text.lower()
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Calculate pattern matches for each document type using pre-compiled patterns
        for doc_type, patterns in self._compiled_patterns.items():
            pattern_matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    pattern_matches += 1
            
            # Calculate keyword matches
            keyword_matches = sum(
                1 for kw in self.DOCUMENT_KEYWORDS.get(doc_type, []) 
                if kw.lower() in text
            )
            
            # Calculate score (weighted sum of pattern and keyword matches)
            # Patterns are weighted more heavily as they're more specific
            pattern_weight = 0.7
            keyword_weight = 0.3
            
            # Normalize by maximum possible matches (to get 0-1 range)
            max_patterns = max(1, len(patterns))
            max_keywords = max(1, len(self.DOCUMENT_KEYWORDS.get(doc_type, [])))
            
            pattern_score = (pattern_matches / max_patterns) * pattern_weight
            keyword_score = (keyword_matches / max_keywords) * keyword_weight
            
            scores[doc_type] = pattern_score + keyword_score
        
        # Calculate unknown score (inverse of max score)
        max_score = max(scores[dt] for dt in DocumentType if dt != DocumentType.UNKNOWN)
        scores[DocumentType.UNKNOWN] = max(0, 1.0 - max_score)
        
        # Normalize all scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
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
