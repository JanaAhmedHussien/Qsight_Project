from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime
import config

class PDFGen:
    def create(self, patient, diag, llm):
        path = config.Config.REPORTS / f"report_{patient.id}.pdf"

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(path), pagesize=letter)

        c = llm  # shorthand

        content = [
            Paragraph("Diabetic Retinal Diagnosis Report", styles["Title"]),
            Paragraph(f"Generated: {datetime.utcnow().isoformat()}", styles["Normal"]),

            Paragraph("Patient Information", styles["Heading2"]),
            Paragraph(f"Name: {patient.name}", styles["Normal"]),
            Paragraph(f"Age: {patient.age}", styles["Normal"]),
            Paragraph(f"Sex: {patient.sex}", styles["Normal"]),
            Paragraph(f"Weight: {patient.weight} kg", styles["Normal"]),
            Paragraph(f"Height: {patient.height} cm", styles["Normal"]),
            Paragraph(f"BMI: {patient.bmi}", styles["Normal"]),

            Paragraph("Diagnosis Findings", styles["Heading2"]),
            Paragraph(f"Left Eye: {diag['retinopathy_left']}", styles["Normal"]),
            Paragraph(f"Right Eye: {diag['retinopathy_right']}", styles["Normal"]),
            Paragraph(f"Confidence: {diag['confidence']}%", styles["Normal"]),
            Paragraph(f"Risk Score: {diag['risk']}", styles["Normal"]),

            Paragraph("Condition & Symptom Overview", styles["Heading2"]),
            Paragraph(c["condition_overview"], styles["Normal"]),

            Paragraph("Patient Assessment", styles["Heading2"]),
            Paragraph(c["patient_assessment"], styles["Normal"]),

            Paragraph("Implications", styles["Heading2"]),
            Paragraph(c["implications"], styles["Normal"]),

            Paragraph("Treatment Plan", styles["Heading2"]),
            Paragraph(c["treatment_plan"], styles["Normal"]),

            Paragraph("Life Impact", styles["Heading2"]),
            Paragraph(c["life_impact"], styles["Normal"]),

            Paragraph("Financial Impact", styles["Heading2"]),
            Paragraph(c["financial_impact"], styles["Normal"]),

            Paragraph("Recovery Projection", styles["Heading2"]),
            Paragraph(c["recovery_projection"], styles["Normal"]),

            Paragraph("Additional Assessments", styles["Heading2"]),
            Paragraph(c["additional_assessments"], styles["Normal"]),

            Paragraph("Compliance Notice", styles["Heading2"]),
            Paragraph(c["compliance_notice"], styles["Italic"]),
        ]

        doc.build(content)
        return str(path)
