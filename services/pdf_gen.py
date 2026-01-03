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
            fontSize=14,
            spaceBefore=20,
            spaceAfter=8,
            fontName="Helvetica-Bold",
            textColor="#2C3E50"
        ))
        
        styles.add(ParagraphStyle(
            name="SubHeader",
            fontSize=11,
            spaceBefore=12,
            spaceAfter=4,
            fontName="Helvetica-Bold",
            textColor="#34495E"
        ))
        
        styles.add(ParagraphStyle(
            name="Body",
            fontSize=10,
            leading=14,
            spaceAfter=6
        ))
        
        styles.add(ParagraphStyle(
            name="BulletItem",
            fontSize=10,
            leading=13,
            leftIndent=10,
            spaceAfter=3,
            bulletIndent=5
        ))
        
        styles.add(ParagraphStyle(
            name="CompactBody",
            fontSize=10,
            leading=12,
            spaceAfter=4
        ))
        
        styles.add(ParagraphStyle(
            name="ItalicNote",
            fontSize=9,
            leading=13,
            textColor="#444444",
            fontName="Helvetica-Oblique"
        ))
        
        # Create document
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
        content.append(Spacer(1, 15))
        
        # ---- Patient Info ----
        content.append(Paragraph("Patient Information", styles["SectionHeader"]))
        content.extend(self._kv_table(styles, {
            "Name": patient.name,
            "Age": f"{patient.age}",
            "Sex": patient.sex,
            "Weight": f"{patient.weight} kg",
            "Height": f"{patient.height} cm",
            "BMI": f"{patient.bmi:.2f}",
            "Insulin": f"{patient.insulin}",
            "Smoker": patient.smoker,
            "Alcohol": patient.alcohol,
            "Vascular Disease": patient.vascular
        }))
        
        # ---- Diagnosis ----
        content.append(Spacer(1, 10))
        content.append(Paragraph("Diagnosis Findings", styles["SectionHeader"]))
        content.extend(self._kv_table(styles, {
            "Left Eye": diag["retinopathy_left"],
            "Right Eye": diag["retinopathy_right"],
            "Confidence": f"{diag['confidence']}%",
            "Risk Score": f"{diag['risk']}"
        }))
        
        # ---- LLM Sections ----
        content.extend(self._section("Condition Overview", c["condition_overview"], styles))
        content.extend(self._section("Patient Assessment", c["patient_assessment"], styles))
        content.extend(self._section("Clinical Implications", c["implications"], styles))
        content.extend(self._section("Treatment Plan", c["treatment_plan"], styles))
        content.extend(self._section("Life Impact", c["life_impact"], styles))
        content.extend(self._section("Financial Considerations", c["financial_impact"], styles))
        content.extend(self._section("Recovery Projection", c["recovery_projection"], styles))
        content.extend(self._section("Recommended Additional Assessments", c["additional_assessments"], styles))
        
        # ---- Compliance ----
        content.append(Paragraph("Important Notice", styles["SectionHeader"]))
        content.append(Paragraph(c["compliance_notice"], styles["ItalicNote"]))
        
        doc.build(content)
        return str(path)
    
    # ---------- Enhanced Helpers ----------
    
    def _kv_table(self, styles, data, columns=2):
        """Create a better looking key-value table"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        
        items = list(data.items())
        table_data = []
        
        # Split items into rows based on columns
        for i in range(0, len(items), columns):
            row = []
            for j in range(columns):
                if i + j < len(items):
                    k, v = items[i + j]
                    row.extend([
                        Paragraph(f"<b>{k}:</b>", styles["Body"]),
                        Paragraph(str(v), styles["Body"])
                    ])
                else:
                    row.extend(["", ""])
            table_data.append(row)
        
        # Create table
        col_widths = [80, 120] * columns  # Adjust widths as needed
        table = Table(table_data, colWidths=col_widths)
        
        # Style the table
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        return [table, Spacer(1, 10)]
    
    def _section(self, title, value, styles):
        """Enhanced section handler with better formatting"""
        blocks = []
        
        # Add section header with spacing
        blocks.append(Paragraph(title, styles["SectionHeader"]))
        
        # Handle different content types
        if isinstance(value, str):
            # Check if it has structured content (like "• Lifestyle: ...")
            if "•" in value or "\n•" in value:
                # Parse structured content
                lines = [line.strip() for line in value.split('\n') if line.strip()]
                
                for line in lines:
                    if line.startswith('•'):
                        # Regular bullet point
                        blocks.append(Paragraph(line, styles["Body"]))
                    elif ':' in line and len(line) < 100:
                        # Sub-header style (e.g., "Lifestyle:")
                        blocks.append(Paragraph(f"<b>{line}</b>", styles["Body"]))
                    else:
                        # Regular text
                        blocks.append(Paragraph(line, styles["Body"]))
            else:
                # Plain text - wrap it
                blocks.append(Paragraph(value, styles["Body"]))
        
        elif isinstance(value, list):
            # List of items
            for item in value:
                if isinstance(item, dict):
                    # Handle dictionary items specially
                    for k, v in item.items():
                        blocks.append(Paragraph(f"<b>{k}:</b> {v}", styles["Body"]))
                else:
                    blocks.append(Paragraph(f"• {item}", styles["Body"]))
        
        elif isinstance(value, dict):
            # Dictionary as key-value pairs
            for k, v in value.items():
                blocks.append(Paragraph(f"<b>{k}:</b> {v}", styles["Body"]))
        
        else:
            blocks.append(Paragraph(str(value), styles["Body"]))
        
        # Add spacing after section
        blocks.append(Spacer(1, 12))
        return blocks

