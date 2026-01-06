# database/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, Patient, Diagnosis
import config

engine = create_engine(config.Config.DB)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

class DB:
    def __init__(self):
        self.s = Session()

    # ---------- CREATE ----------
    def create_patient(self, data):
        p = Patient(**data)
        if p.height and p.weight:
            p.bmi = round(p.weight / ((p.height / 100) ** 2), 2)
        self.s.add(p)
        self.s.commit()
        self.s.refresh(p)
        return p

    def create_diagnosis(self, data):
        # Ensure all required fields have defaults
        clean_data = {
            "patient_id": data.get("patient_id"),
            "retinopathy_left": data.get("retinopathy_left", "Unknown"),
            "retinopathy_right": data.get("retinopathy_right", "Unknown"),
            "confidence": data.get("confidence", 0.0),
            "risk": data.get("risk", 0.0),
            "left_img": data.get("left_img", "left_eye.jpg"),
            "right_img": data.get("right_img", "right_eye.jpg"),
            "json_path": data.get("json_path", ""),
            "pdf_path": data.get("pdf_path", ""),
            "summary": str(data.get("summary", ""))
        }
        
        d = Diagnosis(**clean_data)
        self.s.add(d)
        self.s.commit()
        self.s.refresh(d)
        return d

    # ---------- READ ----------
    def get_patient(self, pid):
        return self.s.query(Patient).filter_by(id=pid).first()

    def get_diagnosis(self, did):
        return self.s.query(Diagnosis).filter_by(id=did).first()

    def list_patients(self):
        return self.s.query(Patient).all()

    def list_diagnoses(self, patient_id=None):
        query = self.s.query(Diagnosis)
        if patient_id:
            query = query.filter_by(patient_id=patient_id)
        return query.all()

    # ---------- UPDATE ----------
    def update_patient(self, pid, updates):
        p = self.get_patient(pid)
        for k, v in updates.items():
            if hasattr(p, k):
                setattr(p, k, v)
        self.s.commit()
        return p

    def update_diagnosis(self, did, updates):
        d = self.get_diagnosis(did)
        for k, v in updates.items():
            if hasattr(d, k):
                setattr(d, k, v)
        self.s.commit()
        return d

    # ---------- DELETE ----------
    def delete_patient(self, pid):
        p = self.get_patient(pid)
        self.s.delete(p)
        self.s.commit()

    def close(self):
        self.s.close()