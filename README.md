# 🌊 MarineOps

> A Decision-Support System for Prioritising Plastic Cleanup Operations Using Calibrated Detection & Severity Ranking.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [API Endpoints](#api-endpoints)
- [PSI Scoring](#psi-scoring)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

MarineOps is a web application that helps marine conservation teams decide **where to send cleanup crews first**. Upload drone or satellite images of a coastal area, and the system will:

1. Detect plastic debris using a YOLOv8 model
2. Calculate a **Plastic Severity Index (PSI)** score per image and per location
3. Rank locations by severity so you know exactly where to act

---

## Features

- 📸 **Image upload** with automatic validation (resolution, size, format)
- 🔍 **YOLOv8 detection** — identifies 9 classes of marine debris
- 📊 **PSI scoring** — per-image and aggregated per-location severity scores
- 🗺️ **Interactive dashboard** — charts, maps, and trend analysis via Plotly Dash
- 📋 **Detection history** — browse all past scans, filter by location
- 📄 **Export reports** — download results as PDF or CSV
- ⚙️ **Model calibration** — isotonic regression to improve confidence accuracy

---

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **PostgreSQL 14+** — [postgresql.org](https://www.postgresql.org/download/)
- **pgAdmin 4** (optional, for database inspection) — [pgadmin.org](https://www.pgadmin.org/)
- **Git** — [git-scm.com](https://git-scm.com/)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/marineops.git
cd marineops
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn[standard] sqlalchemy ultralytics scikit-learn \
            reportlab numpy opencv-python pandas python-dotenv \
            python-multipart httpx pillow dash plotly psycopg2-binary jinja2
```

### 4. Create the PostgreSQL database

Open pgAdmin or psql and run:

```sql
CREATE DATABASE marineops;
```

### 5. Place your model file

Copy your trained YOLOv8 weights into the project root:

```
marineops/
└── best.pt   ← place here
```

---

## Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/marineops
```

Replace `yourpassword` with your PostgreSQL password.

---

## Running the App

### 1. Initialise the database tables

Run this once before the first launch:

```bash
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```

### 2. Start the server

```bash
uvicorn main:app --reload --port 8000
```

### 3. Open the app

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Main application |
| http://localhost:8000/docs | API documentation (Swagger UI) |
| http://localhost:8000/dashboard | Plotly Dash analytics |

---

## Usage Guide

### Uploading and scanning an image

1. Go to the **Upload** page
2. Select an image file (JPEG, PNG — min 320×320px, max 20MB)
3. Enter the **location name** (e.g. `Choisy`, `Grand Baie`)
4. Click **Detect** — results appear within seconds

### Viewing history

Go to the **History** page to see:
- **Photo Upload History** — one row per image, with PSI score, object count, confidence, and timestamp
- **Location History** — one row per location, with aggregated PSI, total objects, and severity

Use the **Filter by location** box to search. Clearing it shows all records.

### Exporting reports

From the **History** page, click:
- **⬇ CSV** — downloads a spreadsheet of all detections
- **⬇ PDF** — downloads a formatted report

### Calibrating the model

If you have a labelled validation dataset, run calibration to improve confidence accuracy:

```bash
curl -X POST http://localhost:8000/calibrate
```

Or use the **Calibrate** button in the UI. The calibrator is saved automatically and loaded on every server restart.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/validate` | Validate an image before detection |
| `POST` | `/detect` | Run detection on an uploaded image |
| `POST` | `/calibrate` | Train the confidence calibrator |
| `GET` | `/analytics` | Get aggregated statistics (supports `?location=` filter) |
| `GET` | `/report/csv` | Download CSV report |
| `GET` | `/report/pdf` | Download PDF report |

Full interactive docs at **http://localhost:8000/docs**.

---

## PSI Scoring

The **Plastic Severity Index** measures how polluted a scanned area is.

### Formula

```
PSI (per object) = class_weight × calibrated_confidence² × (object_area / image_area)
PSI (per image)  = Σ(PSI per object) × 100
PSI (per location) = Σ(PSI per image) for all images at that location
```

### Class weights

| Class | Weight |
|-------|--------|
| `fishing_net` | 2.0 |
| `buoy` | 2.0 |
| `other_fishing_gear` | 2.0 |
| `pet_bottle` | 1.5 |
| `other_bottle` | 1.5 |
| `other_container` | 1.5 |
| `styrene_foam` | 1.2 |
| `fragment` | 1.0 |

### Severity levels

| PSI | Severity |
|-----|----------|
| < 0.5 | 🟢 Low |
| 0.5 – 1.49 | 🟡 Moderate |
| 1.5 – 2.99 | 🟠 High |
| ≥ 3.0 | 🔴 Critical |

---

## Project Structure

```
marineops/
├── main.py               # FastAPI app — all endpoints
├── database.py           # SQLAlchemy models and DB session
├── dashboard.py          # Plotly Dash app
├── best.pt               # YOLOv8 model weights
├── calibrator.pkl        # Saved calibrator (auto-generated)
├── .env                  # Environment variables (not committed)
├── uploads/              # Uploaded images (auto-created)
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── assets/
│       └── logo.png
└── templates/
    └── index.html
```

---

## Troubleshooting

**Server won't start**
```bash
# Make sure you're in the virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

# Then start
uvicorn main:app --reload --port 8000
```

**Database connection error**
- Check your `.env` file has the correct password
- Make sure PostgreSQL is running
- Confirm the `marineops` database exists in pgAdmin

**`best.pt` not found**
- Place your YOLOv8 weights file in the project root and name it `best.pt`

**Port already in use**
```bash
uvicorn main:app --reload --port 8080
# Then visit http://localhost:8080
```

**Static files returning 404**
- Make sure the `static/` folder exists in the project root
- Check that `app.mount("/static", ...)` appears before the dashboard mount in `main.py`

---

## License

This project was developed as part of an undergraduate research project at Middlesex University.
