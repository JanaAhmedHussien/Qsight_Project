import json
from datetime import datetime
import config

class JSONGen:
    def create(self, patient, diag):
        path = config.Config.JSON / f"diagnosis_{patient.id}.json"

        data = {
            "metadata": {
                "system": "QSight",
                "version": "1.0",
                "generated_at": datetime.utcnow().isoformat()
            },
            "patient_info": {
                "id": patient.id,
                "name": patient.name,
                "age": patient.age,
                "sex": patient.sex,
                "bmi": patient.bmi,
                "vascular_risk": patient.vascular
            },
            "image_info": {
                "left_eye": diag["left_img"],
                "right_eye": diag["right_img"],
                "resolution": "224x224"
            },
            "model_info": {
                "model": "DR-CNN",
                "framework": "PyTorch",
                "task": "Diabetic Retinopathy Classification"
            },
            "diagnosis_info": {
                "stage": diag["retinopathy"],
                "confidence": diag["confidence"],
                "risk_score": diag["risk"]
            },
            "llm_narrative_summary": diag["summary"]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return str(path)
