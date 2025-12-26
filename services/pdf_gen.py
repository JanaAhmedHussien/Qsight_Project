from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime
import config

class PDFGen:
    def create(self, p, d):
        path = config.Config.REPORTS / f"report_{p.id}.pdf"

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(path), pagesize=letter)

        doc.build([
            Paragraph("QSight Retinal Diagnosis Report", styles["Title"]),
            Paragraph(f"Generated: {datetime.utcnow()}", styles["Normal"]),
            Paragraph(f"Patient ID: {p.id}", styles["Normal"]),
            Paragraph(f"Diagnosis: {d['retinopathy']}", styles["Normal"]),
            Paragraph(f"Confidence: {d['confidence']}%", styles["Normal"]),
            Paragraph(d["summary"], styles["Normal"]),
            Paragraph("AI-assisted report. Not a medical diagnosis.", styles["Italic"]),
        ])

        return str(path)
