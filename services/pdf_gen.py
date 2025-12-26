from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime
import config

class PDFGen:
    def create(self, patient, diag):
        path = config.Config.REPORTS / f"report_{patient.id}.pdf"

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(path), pagesize=letter)

        content = [
            Paragraph("QSight Retinal Diagnosis Report", styles["Title"]),
            Paragraph(f"Generated: {datetime.utcnow()}", styles["Normal"]),

            Paragraph("<b>1. Executive Summary</b>", styles["Heading2"]),
            Paragraph(diag["summary"], styles["Normal"]),

            Paragraph("<b>2. Patient Information</b>", styles["Heading2"]),
            Paragraph(f"Patient ID: {patient.id}", styles["Normal"]),
            Paragraph(f"Age / Sex: {patient.age} / {patient.sex}", styles["Normal"]),
            Paragraph(f"BMI: {patient.bmi}", styles["Normal"]),

            Paragraph("<b>3. Diagnosis Findings</b>", styles["Heading2"]),
            Paragraph(f"Stage: {diag['retinopathy']}", styles["Normal"]),
            Paragraph(f"Confidence: {diag['confidence']}%", styles["Normal"]),
            Paragraph(f"Risk Score: {diag['risk']}", styles["Normal"]),

            Paragraph("<b>4. Compliance Notice</b>", styles["Heading2"]),
            Paragraph(
                "This AI-generated report is intended for clinical decision support "
                "and must be reviewed by a certified ophthalmologist.",
                styles["Italic"]
            ),
        ]

        doc.build(content)
        return str(path)
