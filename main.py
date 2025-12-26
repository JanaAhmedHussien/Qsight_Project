from database.db import DB
from services.llm import LLM
from services.json_gen import JSONGen
from services.pdf_gen import PDFGen

db = DB()
llm = LLM()
jg = JSONGen()
pg = PDFGen()

# CREATE PATIENT
patient = db.create_patient({
    "name": "Test Patient",
    "age": 45,
    "sex": "M",
    "weight": 75,
    "height": 175,
    "insulin": 12,
    "smoker": False,
    "alcohol": "Low",
    "vascular": False
})

# DIAGNOSIS DATA
diag = {
    "retinopathy": "Mild",
    "confidence": 85.5,
    "risk": 4.0,
    "left_img": "left_eye.jpg",
    "right_img": "right_eye.jpg"
}

# LLM SUMMARY
diag["summary"] = llm.summarize(diag)

# JSON + PDF
diag["json_path"] = jg.create(patient, diag)
diag["pdf_path"] = pg.create(patient, diag)
diag["patient_id"] = patient.id

# SAVE DIAGNOSIS
db.create_diagnosis(diag)

db.close()
print(" QSight pipeline executed successfully")
