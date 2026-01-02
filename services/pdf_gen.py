from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT
from datetime import datetime
import config


class PDFGen:
    def create(self, patient, diag, llm):
        path = config.Config.REPORTS / f"report_{patient.id}.pdf"

        styles = getSampleStyleSheet()

        # ---- Custom styles ----
        styles.add(ParagraphStyle(
            name="Meta",
            fontSize=9,
            textColor="#555555",
            spaceAfter=12
        ))

        styles.add(ParagraphStyle(
            name="SectionHeader",
            fontSize=13,
            spaceBefore=18,
            spaceAfter=6,
            fontName="Helvetica-Bold"
        ))

        styles.add(ParagraphStyle(
            name="Body",
            fontSize=10,
            leading=14,
            spaceAfter=8
        ))

        styles.add(ParagraphStyle(
            name="Label",
            fontSize=10,
            fontName="Helvetica-Bold"
        ))

        styles.add(ParagraphStyle(
            name="ItalicNote",
            fontSize=9,
            leading=13,
            textColor="#444444",
            fontName="Helvetica-Oblique"
        ))

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        c = llm  # shorthand
        content = []

        # ---- Title ----
        content.append(Paragraph("Diabetic Retinal Diagnosis Report", styles["Title"]))
        content.append(Paragraph(
            f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Meta"]
        ))

        # ---- Patient Info ----
        content.append(Paragraph("Patient Information", styles["SectionHeader"]))
        content.extend(self._kv(styles, {
            "Name": patient.name,
            "Age": patient.age,
            "Sex": patient.sex,
            "Weight": f"{patient.weight} kg",
            "Height": f"{patient.height} cm",
            "BMI": patient.bmi
        }))

        # ---- Diagnosis ----
        content.append(Spacer(1, 10))
        content.append(Paragraph("Diagnosis Findings", styles["SectionHeader"]))
        content.extend(self._kv(styles, {
            "Left Eye": diag["retinopathy_left"],
            "Right Eye": diag["retinopathy_right"],
            "Confidence": f"{diag['confidence']}%",
            "Risk Score": diag["risk"]
        }))

        # ---- LLM Sections ----
        content.extend(self._section("Condition Overview", c["condition_overview"], styles))
        content.extend(self._section("Patient Assessment", c["patient_assessment"], styles))
        content.extend(self._section("Implications", c["implications"], styles))
        content.extend(self._section("Treatment Plan", c["treatment_plan"], styles))
        content.extend(self._section("Life Impact", c["life_impact"], styles))
        content.extend(self._section("Financial Impact", c["financial_impact"], styles))
        content.extend(self._section("Recovery Projection", c["recovery_projection"], styles))
        content.extend(self._section("Additional Assessments", c["additional_assessments"], styles))

        # ---- Compliance ----
        content.append(Paragraph("Compliance Notice", styles["SectionHeader"]))
        content.append(Paragraph(c["compliance_notice"], styles["ItalicNote"]))

        doc.build(content)
        return str(path)

    # ---------- Helpers ----------

    def _kv(self, styles, data):
        """Key-value layout"""
        blocks = []
        for k, v in data.items():
            blocks.append(
                Paragraph(f"<b>{k}:</b> {v}", styles["Body"])
            )
        return blocks

    def _section(self, title, value, styles, italic=False):
        body_style = styles["ItalicNote"] if italic else styles["Body"]
        blocks = [Paragraph(title, styles["SectionHeader"])]

        if isinstance(value, str):
            if "•" in value:
                items = [i.strip() for i in value.split("•") if i.strip()]
                blocks.append(ListFlowable(
                    [ListItem(Paragraph(i, body_style)) for i in items],
                    bulletType="bullet",
                    leftIndent=18
                ))
            else:
                blocks.append(Paragraph(value, body_style))

        elif isinstance(value, list):
            blocks.append(ListFlowable(
                [ListItem(Paragraph(str(i), body_style)) for i in value],
                bulletType="bullet",
                leftIndent=18
            ))

        elif isinstance(value, dict):
            items = [f"<b>{k}:</b> {v}" for k, v in value.items()]
            blocks.append(ListFlowable(
                [ListItem(Paragraph(i, body_style)) for i in items],
                bulletType="bullet",
                leftIndent=18
            ))

        else:
            blocks.append(Paragraph(str(value), body_style))

        return blocks


