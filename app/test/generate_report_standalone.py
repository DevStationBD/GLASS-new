#!/usr/bin/env python3
"""
Standalone Report Generator

Generate PDF or HTML reports from existing GLASS tracking JSON files.
This script can be run independently without the GUI to avoid Qt conflicts.
"""

import os
import sys
from pathlib import Path
import argparse

def find_latest_tracking_report():
    """Find the most recent tracking report JSON file"""
    output_dir = Path("output")
    if not output_dir.exists():
        return None
    
    reports = []
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            for session_dir in model_dir.iterdir():
                if session_dir.is_dir():
                    report_dir = session_dir / "report"
                    if report_dir.exists():
                        tracking_reports = list(report_dir.glob("*_tracking_report.json"))
                        for report in tracking_reports:
                            reports.append((report, report.stat().st_mtime))
    
    if not reports:
        return None
    
    # Return the most recent report
    reports.sort(key=lambda x: x[1], reverse=True)
    return str(reports[0][0])

def generate_pdf_report(json_path: str, open_after: bool = True):
    """Generate PDF report"""
    try:
        from app.pdf_report_generator import GLASSReportGenerator
        
        print(f"📄 Generating PDF report from: {json_path}")
        generator = GLASSReportGenerator()
        pdf_path = generator.generate_report(json_path, open_after=open_after)
        
        print(f"✅ PDF report generated successfully!")
        print(f"📁 Location: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("⚠️  ReportLab not installed. Install with: pip install reportlab")
        return None
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return None

def generate_html_report(json_path: str, open_after: bool = True):
    """Generate HTML report"""
    try:
        from app.html_report_generator import GLASSHTMLReportGenerator
        
        print(f"📄 Generating HTML report from: {json_path}")
        generator = GLASSHTMLReportGenerator()
        html_path = generator.generate_report(json_path, open_after=open_after)
        
        print(f"✅ HTML report generated successfully!")
        print(f"📁 Location: {html_path}")
        return html_path
        
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
        return None

def list_available_reports():
    """List all available tracking reports"""
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ No output directory found")
        return []
    
    reports = []
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            for session_dir in model_dir.iterdir():
                if session_dir.is_dir():
                    report_dir = session_dir / "report"
                    if report_dir.exists():
                        tracking_reports = list(report_dir.glob("*_tracking_report.json"))
                        for report in tracking_reports:
                            reports.append(report)
    
    if reports:
        print(f"📋 Found {len(reports)} tracking report(s):")
        for i, report in enumerate(sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True), 1):
            mtime = report.stat().st_mtime
            import datetime
            time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {i}. {report} (modified: {time_str})")
    else:
        print("❌ No tracking reports found")
    
    return reports

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate reports from GLASS tracking JSON files')
    parser.add_argument('--json', '-j', help='Path to specific JSON tracking report')
    parser.add_argument('--latest', '-l', action='store_true', help='Use latest tracking report')
    parser.add_argument('--list', action='store_true', help='List available tracking reports')
    parser.add_argument('--format', '-f', choices=['pdf', 'html', 'both'], default='both',
                       help='Report format (default: both)')
    parser.add_argument('--no-open', action='store_true', help='Do not open reports after generation')
    
    args = parser.parse_args()
    
    print("🔍 GLASS Standalone Report Generator")
    print("=" * 50)
    
    # List reports if requested
    if args.list:
        list_available_reports()
        return
    
    # Determine JSON file to use
    json_path = None
    
    if args.json:
        json_path = args.json
        if not os.path.exists(json_path):
            print(f"❌ JSON file not found: {json_path}")
            return
    elif args.latest:
        json_path = find_latest_tracking_report()
        if not json_path:
            print("❌ No tracking reports found")
            print("💡 Run an inference first or use --list to see available reports")
            return
        print(f"📄 Using latest report: {json_path}")
    else:
        # Interactive selection
        reports = list_available_reports()
        if not reports:
            return
        
        try:
            choice = input(f"\nEnter report number (1-{len(reports)}) or press Enter for latest: ").strip()
            if not choice:
                json_path = str(sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[0])
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(reports):
                    json_path = str(sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[idx])
                else:
                    print("❌ Invalid choice")
                    return
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input")
            return
    
    print(f"\n📊 Generating report(s) from: {os.path.basename(json_path)}")
    print("-" * 50)
    
    open_after = not args.no_open
    success = False
    
    # Generate reports based on format choice
    if args.format in ['pdf', 'both']:
        pdf_path = generate_pdf_report(json_path, open_after)
        if pdf_path:
            success = True
    
    if args.format in ['html', 'both']:
        html_path = generate_html_report(json_path, open_after)
        if html_path:
            success = True
    
    if success:
        print(f"\n✅ Report generation completed!")
        if not open_after:
            print("💡 Reports saved to Documents folder")
    else:
        print(f"\n❌ Report generation failed")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Report generation cancelled")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
