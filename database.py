from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String, index=True)
    psi_score = Column(Float)
    severity = Column(String)
    total_objects = Column(Integer)
    class_counts = Column(Text)
    avg_confidence = Column(Float, nullable=True)
    image_path = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class LocationSummary(Base):
    __tablename__ = "location_summary"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String, unique=True, index=True)
    avg_psi = Column(Float)
    total_scans = Column(Integer)
    last_updated = Column(DateTime)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
