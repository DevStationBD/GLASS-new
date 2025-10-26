#!/usr/bin/env python3
"""
PDF Report Generator for GLASS Inference Results

Converts JSON tracking reports into professional PDF reports with tables and charts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import platform

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not installed. Install with: pip install reportlab")


class GLASSReportGenerator:
    """Generate professional PDF reports from GLASS inference results"""
    
    def __init__(self):
        self.styles = None
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            
            # Custom styles
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center
                textColor=colors.darkblue
            )
            
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            self.summary_style = ParagraphStyle(
                'SummaryStyle',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                leftIndent=20
            )
    
    def load_report_data(self, json_path: str) -> dict:
        """Load tracking report from JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Report file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def create_summary_section(self, data: dict) -> list:
        """Create summary section with key metrics"""
        elements = []
        
        # Title
        elements.append(Paragraph("GLASS Fabric Inspection Report", self.title_style))
        elements.append(Spacer(1, 20))
        
        # Session Information
        elements.append(Paragraph("Session Information", self.heading_style))
        
        session_info = [
            ["Session ID", data.get('session_id', 'N/A')],
            ["Timestamp", data.get('timestamp', 'N/A')],
            ["Video File", data.get('video_file', 'N/A')],
            ["Output File", data.get('output_file', 'N/A')]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(session_table)
        elements.append(Spacer(1, 20))
        
        # Processing Statistics
        elements.append(Paragraph("Processing Statistics", self.heading_style))
        
        processing_stats = [
            ["Frames Processed", f"{data.get('frames_processed', 0):,}"],
            ["Processing Time", f"{data.get('processing_time_seconds', 0):.2f} seconds"],
            ["Processing FPS", f"{data.get('fps_processing', 0):.1f}"],
            ["Unique Defects Found", f"{data.get('unique_defects', 0)}"],
            ["Peak Concurrent Defects", f"{data.get('peak_concurrent_defects', 0)}"]
        ]
        
        processing_table = Table(processing_stats, colWidths=[2*inch, 4*inch])
        processing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(processing_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_tracking_statistics_section(self, data: dict) -> list:
        """Create tracking statistics section"""
        elements = []
        
        tracking_stats = data.get('tracking_statistics', {})
        if not tracking_stats:
            return elements
        
        elements.append(Paragraph("Tracking Statistics", self.heading_style))
        
        # Fabric Motion Analysis
        fabric_stats = [
            ["Total Detections", f"{tracking_stats.get('total_detections', 0):,}"],
            ["Avg Detections per Frame", f"{tracking_stats.get('avg_detections_per_frame', 0):.2f}"],
            ["Current Fabric Speed", f"{tracking_stats.get('current_fabric_speed', 0):.2f} mm/s"],
            ["Fabric State", tracking_stats.get('fabric_state', 'unknown')],
            ["Speed Confidence", f"{tracking_stats.get('speed_confidence', 0):.3f}"],
            ["Speed Bootstrapped", "Yes" if tracking_stats.get('is_speed_bootstrapped', False) else "No"]
        ]
        
        fabric_table = Table(fabric_stats, colWidths=[2*inch, 4*inch])
        fabric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(fabric_table)
        elements.append(Spacer(1, 20))
        
        # Defect Summary
        defect_summary = [
            ["Total Tracks Created", f"{tracking_stats.get('total_tracks_created', 0)}"],
            ["Completed Tracks", f"{tracking_stats.get('completed_tracks', 0)}"],
            ["Total Defect Area", f"{tracking_stats.get('total_defect_area_mm2', 0):.2f} mm²"],
            ["Avg Defect Duration", f"{tracking_stats.get('avg_defect_duration', 0):.1f} frames"],
            ["Max Defect Duration", f"{tracking_stats.get('max_defect_duration', 0)} frames"]
        ]
        
        defect_table = Table(defect_summary, colWidths=[2*inch, 4*inch])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(defect_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_defects_table(self, data: dict) -> list:
        """Create detailed defects table"""
        elements = []
        
        tracked_defects = data.get('tracked_defects', [])
        if not tracked_defects:
            elements.append(Paragraph("No defects were tracked in this session.", self.summary_style))
            return elements
        
        elements.append(Paragraph("Detailed Defect Analysis", self.heading_style))
        
        # Table headers
        headers = [
            "Track ID",
            "Defect Type", 
            "First Frame",
            "Last Frame",
            "Duration\n(frames)",
            "Max Confidence",
            "Max Area\n(mm²)",
            "Trajectory\nLength"
        ]
        
        # Table data
        table_data = [headers]
        
        for defect in tracked_defects:
            row = [
                str(defect.get('track_id', '')),
                defect.get('defect_type', ''),
                str(defect.get('first_frame', '')),
                str(defect.get('last_frame', '')),
                str(defect.get('duration_frames', '')),
                f"{defect.get('max_confidence', 0):.3f}",
                f"{defect.get('max_area_mm2', 0):.2f}",
                str(defect.get('trajectory_length', ''))
            ]
            table_data.append(row)
        
        # Create table
        defects_table = Table(table_data, colWidths=[
            0.7*inch,  # Track ID
            1.2*inch,  # Defect Type
            0.8*inch,  # First Frame
            0.8*inch,  # Last Frame
            0.8*inch,  # Duration
            1.0*inch,  # Max Confidence
            0.9*inch,  # Max Area
            0.8*inch   # Trajectory Length
        ])
        
        # Table styling
        defects_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(defects_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_defect_types_chart(self, data: dict) -> list:
        """Create defect types distribution chart"""
        elements = []
        
        tracking_stats = data.get('tracking_statistics', {})
        defect_types = tracking_stats.get('defect_types_distribution', {})
        
        if not defect_types:
            return elements
        
        elements.append(Paragraph("Defect Types Distribution", self.heading_style))
        
        # Create pie chart
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 50
        pie.y = 50
        pie.width = 100
        pie.height = 100
        
        # Data for pie chart
        pie.data = list(defect_types.values())
        pie.labels = list(defect_types.keys())
        
        # Colors
        colors_list = [colors.red, colors.blue, colors.green, colors.orange, colors.purple]
        pie.slices.strokeColor = colors.black
        pie.slices.strokeWidth = 0.5
        
        for i, color in enumerate(colors_list[:len(defect_types)]):
            pie.slices[i].fillColor = color
        
        drawing.add(pie)
        elements.append(drawing)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def get_documents_folder(self) -> str:
        """Get the user's Documents folder path"""
        system = platform.system()
        
        if system == "Windows":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                  r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                    documents_path = winreg.QueryValueEx(key, "Personal")[0]
                    return documents_path
            except:
                return os.path.join(os.path.expanduser("~"), "Documents")
        
        elif system == "Darwin":  # macOS
            return os.path.join(os.path.expanduser("~"), "Documents")
        
        else:  # Linux and others
            # Try XDG user dirs first
            try:
                result = subprocess.run(['xdg-user-dir', 'DOCUMENTS'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            
            # Fallback to ~/Documents
            return os.path.join(os.path.expanduser("~"), "Documents")
    
    def open_pdf(self, pdf_path: str):
        """Open PDF file with default system viewer"""
        system = platform.system()
        
        try:
            if system == "Windows":
                os.startfile(pdf_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", pdf_path])
            else:  # Linux and others
                subprocess.run(["xdg-open", pdf_path])
        except Exception as e:
            print(f"Could not open PDF automatically: {e}")
            print(f"PDF saved at: {pdf_path}")
    
    def generate_report(self, json_path: str, output_path: str = None, open_after: bool = True) -> str:
        """Generate PDF report from JSON tracking data"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        # Load data
        data = self.load_report_data(json_path)
        
        # Determine output path
        if output_path is None:
            documents_folder = self.get_documents_folder()
            session_id = data.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            filename = f"GLASS_Report_{session_id}.pdf"
            output_path = os.path.join(documents_folder, filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build content
        elements = []
        
        # Summary section
        elements.extend(self.create_summary_section(data))
        
        # Tracking statistics
        elements.extend(self.create_tracking_statistics_section(data))
        
        # Defects table
        elements.extend(self.create_defects_table(data))
        
        # Defect types chart (if applicable)
        elements.extend(self.create_defect_types_chart(data))
        
        # Footer
        elements.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by GLASS Fabric Inspector"
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        
        print(f"✅ PDF report generated: {output_path}")
        
        # Open PDF if requested
        if open_after:
            self.open_pdf(output_path)
        
        return output_path


def main():
    """Test the report generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PDF report from GLASS tracking JSON')
    parser.add_argument('json_path', help='Path to JSON tracking report')
    parser.add_argument('--output', '-o', help='Output PDF path (optional)')
    parser.add_argument('--no-open', action='store_true', help='Do not open PDF after generation')
    
    args = parser.parse_args()
    
    generator = GLASSReportGenerator()
    
    try:
        output_path = generator.generate_report(
            json_path=args.json_path,
            output_path=args.output,
            open_after=not args.no_open
        )
        print(f"Report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
