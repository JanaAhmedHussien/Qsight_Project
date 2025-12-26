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

    def add(self, obj):
        self.s.add(obj)
        self.s.commit()
        self.s.refresh(obj)
        return obj

    def create_patient(self, d):
        p = Patient(**d)
        if p.height and p.weight:
            p.bmi = round(p.weight / (p.height / 100) ** 2, 2)
        return self.add(p)

    def create_diagnosis(self, d):
        return self.add(Diagnosis(**d))

    def close(self):
        self.s.close()
