import pytest
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

# ── Helpers ──────────────────────────────────────────────
def make_test_image(w=640, h=640):
    """Generate a dummy image for testing"""
    img_array = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

# ── Validate Tests ───────────────────────────────────────
def test_validate_valid_image():
    img = make_test_image(640, 640)
    response = client.post(
        "/validate",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["valid"] == True

def test_validate_too_small():
    img = make_test_image(100, 100)
    response = client.post(
        "/validate",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    assert response.status_code == 422

def test_validate_invalid_file():
    response = client.post(
        "/validate",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

# ── Detect Tests ─────────────────────────────────────────
def test_detect_returns_psi():
    img = make_test_image(640, 640)
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img, "image/jpeg")},
        data={"location": "Test Beach"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "psi_score" in data
    assert "severity" in data
    assert "total_objects" in data
    assert "class_counts" in data
    assert data["location"] == "Test Beach"

def test_detect_severity_label():
    img = make_test_image(640, 640)
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img, "image/jpeg")},
        data={"location": "Test Beach"}
    )
    assert response.json()["severity"] in [
        "Low", "Moderate", "High", "Critical"
    ]

# ── Analytics Tests ──────────────────────────────────────
def test_analytics_returns_data():
    response = client.get("/analytics")
    assert response.status_code == 200

def test_analytics_filter_by_location():
    response = client.get("/analytics?location=Test Beach")
    assert response.status_code == 200

# ── Report Tests ─────────────────────────────────────────
def test_csv_export():
    response = client.get("/report/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

def test_pdf_export():
    response = client.get("/report/pdf")
    assert response.status_code == 200
    assert "application/pdf" in response.headers["content-type"]