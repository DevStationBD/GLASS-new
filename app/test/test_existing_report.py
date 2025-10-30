#!/usr/bin/env python3
"""
Test PDF generation with existing tracking report
"""

import os
from pathlib import Path
from app.pdf_report_generator import GLASSReportGenerator

def find_existing_reports():
    """Find existing tracking reports in the output directory"""
    output_dir = Path("output")
    if not output_dir.exists():
        print("No output directory found")
        return []
    
    reports = []
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            for session_dir in model_dir.iterdir():
                if session_dir.is_dir():
                    report_dir = session_dir / "report"
                    if report_dir.exists():
                        tracking_reports = list(report_dir.glob("*_tracking_report.json"))
                        reports.extend(tracking_reports)
    
    return reports

def main():
    """Test PDF generation with existing reports"""
    print("🔍 Looking for existing tracking reports...")
    
    reports = find_existing_reports()
    
    if not reports:
        print("❌ No existing tracking reports found")
        print("💡 Run an inference first to generate tracking reports")
        return
    
    print(f"✅ Found {len(reports)} tracking report(s):")
    for i, report in enumerate(reports, 1):
        print(f"  {i}. {report}")
    
    # Use the most recent report
    latest_report = sorted(reports, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\n📄 Using latest report: {latest_report}")
    
    try:
        # Generate PDF
        generator = GLASSReportGenerator()
        pdf_path = generator.generate_report(str(latest_report), open_after=True)
        
        print(f"✅ PDF report generated successfully!")
        print(f"📁 Location: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
