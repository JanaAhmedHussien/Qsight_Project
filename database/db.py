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
        d = Diagnosis(**data)
        self.s.add(d)
        self.s.commit()
        self.s.refresh(d)
        return d

    # ---------- READ ----------
    def get_patient(self, pid):
        return self.s.query(Patient).filter_by(id=pid).first()

    def list_patients(self):
        return self.s.query(Patient).all()

    # ---------- UPDATE ----------
    def update_patient(self, pid, updates):
        p = self.get_patient(pid)
        for k, v in updates.items():
            setattr(p, k, v)
        self.s.commit()
        return p

    # ---------- DELETE ----------
    def delete_patient(self, pid):
        p = self.get_patient(pid)
        self.s.delete(p)
        self.s.commit()

    def close(self):
        self.s.close()
