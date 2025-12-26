import json, config
from datetime import datetime

class JSONGen:
    def create(self, p, d):
        path = config.Config.JSON / f"diag_{p.id}.json"
        with open(path, "w") as f:
            json.dump({
                "patient": {"id": p.id, "age": p.age, "sex": p.sex, "bmi": p.bmi},
                "diagnosis": d,
                "time": datetime.utcnow().isoformat()
            }, f, indent=2)
        return str(path)
