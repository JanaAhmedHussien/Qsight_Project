import json
from datetime import datetime
import config

class JSONGen:
    def create(self, patient, diag):
        path = config.Config.JSON / f"diagnosis_{patient.id}.json"

        # Ensure diag has required fields
        diag_data = diag.copy() if isinstance(diag, dict) else {}
        
        # Add default values for missing image fields
        left_img = diag_data.get('left_img', 'left_eye.jpg')
        right_img = diag_data.get('right_img', 'right_eye.jpg')
        
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
                "insulin": patient.insulin,
                "smoker": patient.smoker,
                "alcohol": patient.alcohol,
                "vascular_risk": patient.vascular
            },
            "image_info": {
                "left_eye": left_img,
                "right_eye": right_img,
                "resolution": "224x224"
            },
            "model_info": {
                "model": "DR-CNN",
                "framework": "PyTorch",
                "task": "Diabetic Retinopathy Classification"
            },
            "diagnosis_info": {
                "left_eye": diag_data.get('retinopathy_left', 'Unknown'),
                "right_eye": diag_data.get('retinopathy_right', 'Unknown'),
                "confidence": diag_data.get('confidence', 0.0),
                "risk_score": diag_data.get('risk', 0.0),
                "left_confidence": diag_data.get('left_confidence', 0.0),
                "right_confidence": diag_data.get('right_confidence', 0.0)
            },
            "llm_report": diag_data.get('llm', {})
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return str(path)
