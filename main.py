from database.db import DB
from services.llm import LLM
from services.json_gen import JSONGen
from services.pdf_gen import PDFGen

db = DB()
llm = LLM()
jg = JSONGen()
pg = PDFGen()

def profile_builder(**kwargs):
    return db.create_patient(kwargs)

patient = profile_builder(
    name="Test Patient",
    age=45,
    sex="M",
    weight=75,
    height=175,
    insulin=12,
    smoker=False,
    alcohol="Low",
    vascular=False
)

diag = {
    "retinopathy_left": "Mild",
    "retinopathy_right": "Mild",
    "confidence": 85.5,
    "risk": 4.0,
    "left_img": "left_eye.jpg",
    "right_img": "right_eye.jpg"
}

llm_output = llm.generate_report(patient, diag)
diag["llm"] = llm_output

diag["json_path"] = jg.create(patient, diag)
diag["pdf_path"] = pg.create(patient, diag, llm_output)
diag["patient_id"] = patient.id

db.create_diagnosis(diag)
db.close()

print("QSight diagnosis report generated successfully")
