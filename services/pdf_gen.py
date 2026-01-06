# services/pdf_gen.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY
from datetime import datetime
import config


class PDFGen:
    def create(self, patient, diag, llm):
        path = config.Config.REPORTS / f"report_{patient.id}.pdf"
        
        # Get base stylesheet
        styles = getSampleStyleSheet()
        
        # Create custom styles
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        bullet_style = ParagraphStyle(
            "CustomBullet",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            leftIndent=15,
            spaceAfter=3
        )
        
        meta_style = ParagraphStyle(
            "CustomMeta",
            parent=styles["Normal"],
            fontSize=9,
            textColor="#555555",
            spaceAfter=12
        )
        
        section_header_style = ParagraphStyle(
            "CustomSectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=8,
            textColor="#2C3E50"
        )
        
        subheader_style = ParagraphStyle(
            "CustomSubHeader",
            parent=styles["Normal"],
            fontSize=11,
            spaceBefore=12,
            spaceAfter=4,
            textColor="#34495E",
            fontName="Helvetica-Bold"
        )
        
        italic_note_style = ParagraphStyle(
            "CustomItalicNote",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor="#444444",
            fontName="Helvetica-Oblique"
        )
        
        # Create document
        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        content = []
        
        # Title
        content.append(Paragraph("QSight - Diabetic Retinal Diagnosis Report", styles["Title"]))
        content.append(Paragraph(
            f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            meta_style
        ))
        content.append(Spacer(1, 15))
        
        # Patient Info
        content.append(Paragraph("Patient Information", section_header_style))
        content.extend(self._kv_table(body_style, {
            "Name": patient.name,
            "Age": f"{patient.age}",
            "Sex": patient.sex,
            "Weight": f"{patient.weight} kg",
            "Height": f"{patient.height} cm",
            "BMI": f"{patient.bmi:.2f}" if patient.bmi else "N/A",
            "Insulin": f"{patient.insulin}",
            "Smoker": "Yes" if patient.smoker else "No",
            "Alcohol": patient.alcohol,
            "Vascular Disease": "Yes" if patient.vascular else "No"
        }))
        
        # Diagnosis
        content.append(Spacer(1, 10))
        content.append(Paragraph("Diagnosis Findings", section_header_style))
        content.extend(self._kv_table(body_style, {
            "Left Eye": diag.get("retinopathy_left", "Unknown"),
            "Left Confidence": f"{diag.get('left_confidence', 0):.1f}%",
            "Right Eye": diag.get("retinopathy_right", "Unknown"),
            "Right Confidence": f"{diag.get('right_confidence', 0):.1f}%",
            "Average Confidence": f"{diag.get('confidence', 0):.1f}%",
            "Risk Score": f"{diag.get('risk', 0):.1f}/10"
        }))
        
        # LLM Sections
        sections = [
            ("Condition Overview", "condition_overview"),
            ("Patient Assessment", "patient_assessment"),
            ("Clinical Implications", "implications"),
            ("Treatment Plan", "treatment_plan"),
            ("Life Impact", "life_impact"),
            ("Financial Impact", "financial_impact"),
            ("Recovery Projection", "recovery_projection"),
            ("Additional Assessments", "additional_assessments"),
        ]
        
        for title, key in sections:
            if key in llm and llm[key]:
                content.append(Paragraph(title, section_header_style))
                content.extend(self._format_content(llm[key], body_style, bullet_style))
                content.append(Spacer(1, 10))
        
        # Compliance Notice
        if "compliance_notice" in llm and llm["compliance_notice"]:
            content.append(Paragraph("Important Notice", section_header_style))
            content.append(Paragraph(llm["compliance_notice"], italic_note_style))
        
        # Footer
        content.append(Spacer(1, 20))
        content.append(Paragraph(
            "This report is generated by QSight AI System. For clinical decisions, consult with healthcare professionals.",
            italic_note_style
        ))
        
        doc.build(content)
        return str(path)
    
    def _format_content(self, content, body_style, bullet_style):
        """Format content with proper bullet handling"""
        elements = []
        
        if isinstance(content, str):
            # Clean up the content
            content = content.strip()
            
            # Split by bullet points
            if '•' in content:
                parts = content.split('•')
                for part in parts:
                    part = part.strip()
                    if part:
                        # Check if it's a sub-header
                        if ':' in part and len(part) < 100:
                            elements.append(Paragraph(f"<b>{part}</b>", body_style))
                        else:
                            elements.append(Paragraph(f"• {part}", bullet_style))
            else:
                # No bullets, return as regular paragraph
                elements.append(Paragraph(content, body_style))
        
        elif isinstance(content, list):
            # Handle list content
            for item in content:
                if isinstance(item, dict):
                    for k, v in item.items():
                        elements.append(Paragraph(f"<b>{k}:</b> {v}", body_style))
                else:
                    elements.append(Paragraph(f"• {item}", bullet_style))
        else:
            elements.append(Paragraph(str(content), body_style))
        
        return elements
    
    def _kv_table(self, body_style, data, columns=2):
        """Create key-value table"""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        
        items = list(data.items())
        table_data = []
        
        # Split items into rows
        for i in range(0, len(items), columns):
            row = []
            for j in range(columns):
                if i + j < len(items):
                    k, v = items[i + j]
                    row.extend([
                        Paragraph(f"<b>{k}:</b>", body_style),
                        Paragraph(str(v), body_style)
                    ])
                else:
                    row.extend(["", ""])
            table_data.append(row)
        
        # Create table
        col_widths = [100, 120] * columns
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

