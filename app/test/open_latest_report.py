#!/usr/bin/env python3
"""
Open Latest Report

Simple script to find and open the most recent GLASS report.
"""

import os
import subprocess
from pathlib import Path
import glob

def find_latest_pdf_report():
    """Find the most recent PDF report in Documents folder"""
    documents = Path.home() / "Documents"
    pdf_pattern = str(documents / "GLASS_Report_*.pdf")
    
    pdf_files = glob.glob(pdf_pattern)
    if not pdf_files:
        return None
    
    # Sort by modification time, newest first
    pdf_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return pdf_files[0]

def find_latest_html_report():
    """Find the most recent HTML report in Documents folder"""
    documents = Path.home() / "Documents"
    html_pattern = str(documents / "GLASS_Report_*.html")
    
    html_files = glob.glob(html_pattern)
    if not html_files:
        return None
    
    # Sort by modification time, newest first
    html_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return html_files[0]

def open_file_with_command(file_path, commands):
    """Try to open file with different commands"""
    for cmd in commands:
        try:
            subprocess.run([cmd, file_path], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         check=True)
            print(f"✅ Opened with {cmd}: {os.path.basename(file_path)}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False

def main():
    """Main function"""
    print("🔍 Looking for latest GLASS reports...")
    
    # Try to find PDF first
    pdf_path = find_latest_pdf_report()
    html_path = find_latest_html_report()
    
    if not pdf_path and not html_path:
        print("❌ No GLASS reports found in Documents folder")
        print("💡 Generate a report first using: python quick_report.py")
        return
    
    success = False
    
    # Try to open PDF first
    if pdf_path:
        print(f"📄 Found PDF: {os.path.basename(pdf_path)}")
        
        # Try different commands to open PDF
        pdf_commands = [
            'xdg-open',      # Linux default
            'evince',        # GNOME PDF viewer
            'okular',        # KDE PDF viewer
            'firefox',       # Firefox browser
            'google-chrome', # Chrome browser
            'chromium',      # Chromium browser
        ]
        
        if open_file_with_command(pdf_path, pdf_commands):
            success = True
        else:
            print(f"⚠️  Could not open PDF automatically")
            print(f"📁 PDF location: {pdf_path}")
            print(f"💡 Try manually: xdg-open '{pdf_path}'")
    
    # Try to open HTML as fallback
    if html_path and not success:
        print(f"🌐 Found HTML: {os.path.basename(html_path)}")
        
        # Try different commands to open HTML
        html_commands = [
            'xdg-open',      # Linux default
            'firefox',       # Firefox browser
            'google-chrome', # Chrome browser
            'chromium',      # Chromium browser
        ]
        
        if open_file_with_command(html_path, html_commands):
            success = True
        else:
            print(f"⚠️  Could not open HTML automatically")
            print(f"📁 HTML location: {html_path}")
            print(f"💡 Try manually: xdg-open '{html_path}'")
    
    if not success:
        print("\n❌ Could not open any reports automatically")
        print("💡 Try opening the files manually from your Documents folder")
        if pdf_path:
            print(f"   PDF: {pdf_path}")
        if html_path:
            print(f"   HTML: {html_path}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
