#!/usr/bin/env python3
"""
Manual PDF Opener

Direct script to open the latest PDF report with multiple fallback methods.
"""

import os
import subprocess
import sys
from pathlib import Path
import glob

def main():
    """Find and open the latest PDF report"""
    
    # Find the latest PDF
    documents = Path.home() / "Documents"
    pdf_pattern = str(documents / "GLASS_Report_*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print("❌ No GLASS PDF reports found in Documents folder")
        return
    
    # Get the most recent PDF
    latest_pdf = max(pdf_files, key=os.path.getmtime)
    print(f"📄 Latest PDF: {os.path.basename(latest_pdf)}")
    
    # List of commands to try
    commands = [
        ['xdg-open', latest_pdf],
        ['evince', latest_pdf],
        ['okular', latest_pdf],
        ['firefox', latest_pdf],
        ['google-chrome', latest_pdf],
        ['chromium-browser', latest_pdf],
        ['chromium', latest_pdf],
    ]
    
    print("🔄 Trying different methods to open PDF...")
    
    for i, cmd in enumerate(commands, 1):
        try:
            print(f"  {i}. Trying {cmd[0]}...", end=" ")
            result = subprocess.run(cmd, 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL,
                                  timeout=5)
            if result.returncode == 0:
                print(f"✅ Success! {result}")
                print(f"📖 PDF opened with {cmd[0]}")
                return
            else:
                print("❌ Failed")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ Not available")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # If all methods failed, provide manual instructions
    print("\n❌ All automatic methods failed")
    print("📁 Manual options:")
    print(f"   1. File manager: nautilus '{documents}'")
    print(f"   2. Terminal: xdg-open '{latest_pdf}'")
    print(f"   3. Browser: firefox '{latest_pdf}'")
    print(f"   4. Direct path: {latest_pdf}")

if __name__ == '__main__':
    main()
