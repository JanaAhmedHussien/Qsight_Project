from sqlalchemy import Column, Integer, String, Float, Boolean, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    sex = Column(String)
    weight = Column(Float)
    height = Column(Float)
    bmi = Column(Float)
    insulin = Column(Float)
    smoker = Column(Boolean)
    alcohol = Column(String)
    vascular = Column(Boolean)

class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer)
    retinopathy = Column(String)
    confidence = Column(Float)
    risk = Column(Float)
    left_img = Column(String)
    right_img = Column(String)
    json_path = Column(String)
    pdf_path = Column(String)
    summary = Column(Text)
