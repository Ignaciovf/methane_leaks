# models/database.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, DECIMAL, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://methane_user:methane_password@localhost:5433/methane_leaks_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class MethaneLeak(Base):
    __tablename__ = "methane_leaks"

    id = Column(Integer, primary_key=True, index=True)
    id_plume = Column(String(255), unique=True, nullable=False)
    source_name = Column(String(255))
    satellite = Column(String(255))
    tile_date = Column(TIMESTAMP)
    lat = Column(DECIMAL(10, 8))
    lon = Column(DECIMAL(11, 8))
    actionable = Column(String(50))
    notified = Column(Boolean)
    country = Column(String(100))
    sector = Column(String(100))
    detection_institution = Column(String(255))
    quantification_institution = Column(String(255))
    tile = Column(String(255))
    ch4_fluxrate = Column(DECIMAL(15, 2))
    ch4_fluxrate_std = Column(DECIMAL(15, 2))
    wind_u = Column(DECIMAL(10, 2))
    wind_v = Column(DECIMAL(10, 2))
    total_emission = Column(DECIMAL(15, 2))
    total_emission_std = Column(DECIMAL(15, 2))
    wind_speed = Column(DECIMAL(10, 2))
    last_update = Column(TIMESTAMP)
    insert_date = Column(TIMESTAMP)
    feedback_operator = Column(String(255))
    feedback_government = Column(String(255))

    # Our additional tracking columns
    contacted_by_methane_leaks = Column(Boolean, default=False)
    contact_date = Column(TIMESTAMP)
    contact_notes = Column(Text)

    # Spatial column
    geom = Column(Geometry('POINT', srid=4326))

    # Tracking columns
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()