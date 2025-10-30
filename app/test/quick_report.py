#!/usr/bin/env python3
"""
Quick Report Generator

Generate a report from the most recent GLASS inference session.
This script bypasses all GUI issues and generates reports directly.
"""

import os
import sys
from pathlib import Path

def find_latest_session():
    """Find the most recent session directory"""
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ No output directory found")
        return None
    
    # Look for model directories
    latest_session = None
    latest_time = 0
    
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            for session_dir in model_dir.iterdir():
                if session_dir.is_dir():
                    # Check if this session has a report
                    report_dir = session_dir / "report"
                    if report_dir.exists():
                        tracking_reports = list(report_dir.glob("*_tracking_report.json"))
                        if tracking_reports:
                            # Use directory modification time
                            mtime = session_dir.stat().st_mtime
                            if mtime > latest_time:
                                latest_time = mtime
                                latest_session = tracking_reports[0]
    
    return latest_session

def generate_html_report(json_path):
    """Generate HTML report (no dependencies)"""
    try:
        from app.html_report_generator import GLASSHTMLReportGenerator
        
        print(f"📄 Generating HTML report from: {json_path}")
        generator = GLASSHTMLReportGenerator()
        html_path = generator.generate_report(str(json_path), open_after=True)
        
        print(f"✅ HTML report generated!")
        print(f"📁 Location: {html_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_pdf_report(json_path):
    """Generate PDF report (requires reportlab)"""
    try:
        from app.pdf_report_generator import GLASSReportGenerator
        
        print(f"📄 Generating PDF report from: {json_path}")
        generator = GLASSReportGenerator()
        pdf_path = generator.generate_report(str(json_path), open_after=True)
        
        print(f"✅ PDF report generated!")
        print(f"📁 Location: {pdf_path}")
        return True
        
    except ImportError:
        print("⚠️  ReportLab not installed for PDF generation")
        print("💡 Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Quick GLASS Report Generator")
    print("=" * 40)
    
    # Find latest session
    latest_report = find_latest_session()
    if not latest_report:
        print("❌ No recent inference sessions found")
        print("💡 Run an inference first to generate tracking data")
        return
    
    print(f"📊 Found latest session: {latest_report}")
    print("-" * 40)
    
    # Try to generate both reports
    html_success = generate_html_report(latest_report)
    pdf_success = generate_pdf_report(latest_report)
    
    if html_success or pdf_success:
        print("\n✅ Report generation completed!")
        print("📂 Reports saved to Documents folder")
        
        if html_success:
            print("🌐 HTML report should open in your browser")
        if pdf_success:
            print("📄 PDF report should open in your default viewer")
    else:
        print("\n❌ Report generation failed")
        print("💡 Check the error messages above for details")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Report generation cancelled")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
