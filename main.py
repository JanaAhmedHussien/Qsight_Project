from database.db import DB
from services.llm import LLM
from services.json_gen import JSONGen
from services.pdf_gen import PDFGen

db, llm, jg, pg = DB(), LLM(), JSONGen(), PDFGen()

patient = db.create_patient({
    "name":"Test", "age":45, "sex":"M",
    "weight":75, "height":175,
    "insulin":12, "smoker":False,
    "alcohol":"Low", "vascular":False
})

diag = {
    "retinopathy":"Mild",
    "confidence":85.5,
    "risk":4.0,
    "left_img":"L.jpg",
    "right_img":"R.jpg"
}

diag["summary"] = llm.summarize(diag)
diag["json_path"] = jg.create(patient, diag)
diag["pdf_path"] = pg.create(patient, diag)
diag["patient_id"] = patient.id

db.create_diagnosis(diag)
db.close()
print("Pipeline complete")
