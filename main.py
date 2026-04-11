# main.py
# MarineOps Backend API
#
# This file implements the core backend logic for:
# - Image validation
# - Object detection (YOLO)
# - Confidence calibration
# - PSI (Plastic Severity Index) computation
# - Database storage and analytics

# IMPORTS
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from ultralytics import YOLO
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from datetime import datetime
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
import pickle
import shutil
import glob
import json
import io
import os
import cv2
import pandas as pd
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

# SETUP

load_dotenv()

app = FastAPI(title="Marine Debris Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



# Load model and calibrator once at startup
model = YOLO('best.pt')

# Confidence calibration model (optional)
# Improves reliability of predicted confidence scores
# If unavailable, system falls back to raw confidence
calibrator = None
if os.path.exists('calibrator.pkl'):
    with open('calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)

# Class-specific weights for PSI computation
PSI_WEIGHTS = {
    'fishing_net': 2.0, 'buoy': 2.0, 'other_fishing_gear': 2.0,
    'pet_bottle': 1.5, 'other_bottle': 1.5, 'box_shaped_case': 1.5,
    'other_container': 1.5, 'styrene_foam': 1.2, 'fragment': 1.0
}

os.makedirs('uploads', exist_ok=True)


# DATABASE

from database import get_db, Detection, LocationSummary


# HELPERS
def get_severity_label(psi):
    if psi < 0.5: return "Low"
    elif psi < 1.5: return "Moderate"
    elif psi < 3.0: return "High"
    else: return "Critical"

#Applying Isotonic regression ito model confidence scores
def calibrate_confidence(conf):
    if calibrator:
        return float(calibrator.predict([conf])[0])
    return conf


# ENDPOINTS


# Validate: Ensures uploaded image meets system requirements (improves robustness and avoids runtime errors by preventing invalid inputs)
@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = img.shape[:2]
    size_mb = len(contents) / (1024 * 1024)

    errors = []
    if w < 320 or h < 320:
        errors.append(f"Image too small ({w}x{h}). Minimum 320x320.")
    if size_mb > 20:
        errors.append(f"Image too large ({size_mb:.1f}MB). Maximum 20MB.")
    if w / h > 5 or h / w > 5:
        errors.append("Image aspect ratio too extreme.")

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return {
        "valid": True,
        "width": w,
        "height": h,
        "size_mb": round(size_mb, 2)
    }

# Detect
    """
    Core pipeline:
    1. Save uploaded image
    2. Run YOLO detection
    3. Calibrate confidence scores
    4. Compute PSI per object
    5. Aggregate PSI per image
    6. Store results in database
    7. Update location-level statistics
    """
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    location: str = Form(...),
    db: Session = Depends(get_db)
):
    img_path = f"uploads/{file.filename}"
    with open(img_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    results = model(img_path, verbose=False)[0]
    image_h, image_w = results.orig_shape

    detections = []
    total_psi = 0.0
    class_counts = {}
    confidence_scores = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        raw_conf = float(box.conf)
        cal_conf = calibrate_confidence(raw_conf)
    
        confidence_scores.append(cal_conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w_class = PSI_WEIGHTS.get(cls_name, 1.0)
        area = (x2 - x1) * (y2 - y1)
        psi_obj = w_class * (cal_conf ** 2) * (area / (image_w * image_h))
        total_psi += psi_obj
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        detections.append({
            'class': cls_name,
            'confidence': round(cal_conf, 3),
            'psi_contribution': round(psi_obj, 4),
            'bbox': [x1, y1, x2, y2]
        })


    avg_conf = (
        round(sum(confidence_scores)/ len(confidence_scores), 4)
        if confidence_scores else None
    )

    PSI_SCALE_FACTOR = 100
    psi_score = float(round(total_psi * PSI_SCALE_FACTOR, 4))
    severity = get_severity_label(psi_score)
   

    # Save to PostgreSQL
    record = Detection(
        location=location,
        psi_score=float(psi_score),
        severity=severity,
        total_objects=int(len(detections)),
        class_counts=json.dumps(class_counts),
        image_path=img_path,
        avg_confidence =avg_conf,
        timestamp=datetime.utcnow()
    )
    db.add(record)
    db.flush()

        # Get ALL images for this location to recalculate aggregate PSI
    all_location_records = db.query(Detection).filter(
        Detection.location == location
    ).all()
    
    n_images = len(all_location_records)
    total_psi_all = sum(r.psi_score for r in all_location_records)
    
    # PSI_location = Σ(PSI_object) / √n_images
    aggregate_psi = float(round(
        total_psi_all / np.sqrt(n_images), 4
    ))
    aggregate_severity = get_severity_label(aggregate_psi)
    
    # Update location summary
    summary = db.query(LocationSummary).filter(
        LocationSummary.location == location
    ).first()
    
    if summary:
        summary.avg_psi = aggregate_psi
        summary.total_scans = n_images
        summary.last_updated = datetime.utcnow()
    else:
        summary = LocationSummary(
            location=location,
            avg_psi=aggregate_psi,
            total_scans=1,
            last_updated=datetime.utcnow()
        )
        db.add(summary)
    
    db.commit()
    
    return {
        'location': location,
        'image_psi': float(psi_score),
        'image_severity': severity,          # PSI for this image
        'location_psi': aggregate_psi,           # aggregate PSI for location
        'location_severity': aggregate_severity,
        'total_images_at_location': n_images,
        'total_objects': int(len(detections)),
        'class_counts': class_counts,
        'detections': detections
    }

# Calibrate
@app.post("/calibrate")
async def calibrate():
    """
    Train isotonic regression model for confidence calibration.

    Uses validation predictions to map:
    raw confidence → calibrated probability

    Evaluated using Brier score:
    - lower = better calibration

    This improves reliability of downstream PSI scoring.
    """
    global calibrator

    val_images = glob.glob(
        'C:/Users/Micha/OneDrive - Middlesex University/UG Project/Claude/bepli_yolo_v1/images/val'
    )

    if not val_images:
        raise HTTPException(
            status_code=404,
            detail="Validation images not found"
        )

    raw_confs = []
    correct = []

    for img_path in val_images:
        results = model(img_path, conf=0.001, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf)
            raw_confs.append(conf)
            correct.append(1 if conf > 0.5 else 0)

    if len(raw_confs) < 10:
        raise HTTPException(
            status_code=400,
            detail="Not enough detections for calibration"
        )

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_confs, correct)

    cal_confs = calibrator.predict(raw_confs)
    brier = brier_score_loss(correct, cal_confs)

    with open('calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)

    return {
        "status": "Calibration complete",
        "samples_used": len(raw_confs),
        "brier_score": round(brier, 4)
    }

# Analytics
@app.get("/analytics")
def get_analytics(
    location: str = None,
    db: Session = Depends(get_db)
):
    # Base filter applied to ALL queries
    def apply_filter(q):
        if location:
            return q.filter(Detection.location.ilike(f"%{location}%"))
        return q

    # Records
    records = apply_filter(
        db.query(Detection)
    ).order_by(Detection.timestamp.desc()).all()

    if not records:
        return {"message": "No data found"}

    # Class distribution
    total_class_counts = Counter()
    for r in records:
        total_class_counts.update(json.loads(r.class_counts))

    # Location stats — filtered
    location_stats = apply_filter(
        db.query(
            Detection.location,
            func.avg(Detection.psi_score).label('avg_psi'),
            func.max(Detection.psi_score).label('max_psi'),
            func.min(Detection.psi_score).label('min_psi'),
            func.count(Detection.id).label('total_scans'),
            func.sum(Detection.total_objects).label('total_objects')
        )
    ).group_by(Detection.location).all()

    # Severity counts — filtered
    severity_counts = apply_filter(
        db.query(
            Detection.severity,
            func.count(Detection.id).label('count')
        )
    ).group_by(Detection.severity).all()

    # PSI trend — filtered
    psi_trend = apply_filter(
        db.query(
            Detection.timestamp,
            Detection.psi_score,
            Detection.location,
            Detection.total_objects,
            Detection.avg_confidence
        )
    ).order_by(Detection.timestamp).all()

    return {
        "total_scans":            len(records),
        "total_objects_detected": sum(r.total_objects for r in records),
        "class_distribution":     dict(total_class_counts),
        "location_stats": [
            {
                "location":      s.location,
                "avg_psi":       round(s.avg_psi, 4),
                "max_psi":       round(s.max_psi, 4),
                "min_psi":       round(s.min_psi, 4),
                "total_scans":   s.total_scans,
                "total_objects": s.total_objects
            }
            for s in location_stats
        ],
        "severity_distribution": {
            s.severity: s.count for s in severity_counts
        },
        "psi_trend": [
            {
                "timestamp":      t.timestamp.isoformat(),
                "psi_score":      t.psi_score,
                "location":       t.location,
                "total_objects":  t.total_objects or 0,
                "avg_confidence": round(t.avg_confidence, 4) if t.avg_confidence else None,
            }
            for t in psi_trend
        ]
    }
# Report CSV
@app.get("/report/csv")
def export_csv(
    location: str = None,
    db: Session = Depends(get_db)
):
    query = db.query(Detection)
    if location:
        query = query.filter(Detection.location == location)

    records = query.all()
    df = pd.DataFrame([{
        'location': r.location,
        'psi_score': r.psi_score,
        'severity': r.severity,
        'total_objects': r.total_objects,
        'class_counts': r.class_counts,
        'timestamp': r.timestamp
    } for r in records])

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=marine_debris_report.csv"
        }
    )

# Report PDF
@app.get("/report/pdf")
def export_pdf(
    location: str = None,
    db: Session = Depends(get_db)
):
    query = db.query(Detection)
    if location:
        query = query.filter(Detection.location == location)

    records = query.all()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "Marine Debris Detection Report",
        styles['Title']
    ))

    data = [['Location', 'PSI Score', 'Severity',
             'Objects', 'Timestamp']]
    for r in records:
        data.append([
            r.location,
            str(round(r.psi_score, 4)),
            r.severity,
            str(r.total_objects),
            r.timestamp.strftime('%Y-%m-%d %H:%M')
        ])

    table = Table(data)
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                "attachment; filename=marine_debris_report.pdf"
        }
    )

app.mount("/static", StaticFiles(directory="static"), name="static")


# MOUNT DASH
# Mount Dash application inside FastAPI
# Combines:
# - backend API (FastAPI)
# - interactive frontend (Dash)
from dashboard import app_dash
app.mount('/dashboard', WSGIMiddleware(app_dash.server))

@app.get("/")
async def root():
    return FileResponse("static/templates/index.html")