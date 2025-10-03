import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_classify_kra_document():
    """Test classification of a KRA PIN certificate"""
    kra_text = """
    KENYA REVENUE AUTHORITY
    PERSONAL IDENTIFICATION NUMBER (PIN) CERTIFICATE
    
    CERTIFICATE NO: A12345678X
    
    This is to certify that
    TAXPAYER NAME: JOHN DOE MWANGI
    PIN: A12345678X
    
    TAX OBLIGATIONS:
    - PAYE: ACTIVE
    - VAT: ACTIVE
    
    This certificate is issued under the Tax Procedures Act
    """
    
    response = client.post(
        "/classify-document",
        data={"mode": "text"},
        files={"text": (None, kra_text)},
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == "kra_pin_certificate"
    assert result["confidence"] > 0.7
    assert result["needs_review"] is False

def test_classify_id_document():
    """Test classification of a National ID card"""
    id_text = """
    REPUBLIC OF KENYA
    NATIONAL IDENTITY CARD
    
    ID NO: 12345678
    SURNAME: MWANGI
    OTHER NAMES: JOHN DOE
    DATE OF BIRTH: 01-01-1980
    GENDER: MALE
    DATE OF ISSUE: 01-01-2020
    
    This is to certify that the above is a true copy of the
    National Identity Card issued under the Registration of
    Persons Act (Cap. 107).
    """
    
    response = client.post(
        "/classify-document",
        data={"mode": "text"},
        files={"text": (None, id_text)},
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == "national_id"
    assert result["confidence"] > 0.7
    assert result["needs_review"] is False

def test_classify_unknown_document():
    """Test classification of an unknown document type"""
    unknown_text = "This is just some random text that doesn't match any known document type."
    
    response = client.post(
        "/classify-document",
        data={"mode": "text"},
        files={"text": (None, unknown_text)},
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == "unknown"
    assert result["needs_review"] is True

def test_classify_with_image():
    """Test classification with an image file"""
    # This test would require a sample image file
    # For now, we'll skip it in the automated tests
    pass
