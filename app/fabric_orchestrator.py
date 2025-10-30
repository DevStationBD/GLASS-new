#!/usr/bin/env python3
"""
Fabric Inspector Orchestrator

Terminal-based orchestrator that runs the fabric inspector and handles PDF opening.
This solves the issue where OpenCV GUI applications can't properly spawn other GUI apps.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

class FabricOrchestrator:
    def __init__(self):
        self.fabric_inspector_path = "app/fabric_inspector.py"
        self.reports_generated = []
        
    def print_banner(self):
        """Print welcome banner"""
        print("=" * 60)
        print("🏭 GLASS Fabric Inspector Orchestrator")
        print("=" * 60)
        print("This orchestrator manages the fabric inspector and handles reports.")
        print("When inference completes, reports will be opened automatically.")
        print("=" * 60)
    
    def check_fabric_inspector(self):
        """Check if fabric inspector exists"""
        if not os.path.exists(self.fabric_inspector_path):
            print(f"❌ Fabric inspector not found: {self.fabric_inspector_path}")
            return False
        return True
    
    def open_pdf_report(self, pdf_path):
        """Open PDF report in browser (background)"""
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        print(f"📄 Opening PDF report in browser: {os.path.basename(pdf_path)}")
        
        # Convert to file:// URL for browser
        file_url = f"file://{os.path.abspath(pdf_path)}"
        
        try:
            # Open in background with xdg-open (uses default browser)
            subprocess.Popen(
                ["xdg-open", file_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print("✅ PDF opened in default browser")
            return True
        except Exception as e:
            print(f"❌ Failed to open PDF: {e}")
            print(f"📁 PDF location: {pdf_path}")
            return False

    def find_latest_report(self):
        """Find the most recent PDF report"""
        documents = Path.home() / "Documents"
        pdf_pattern = documents / "GLASS_Report_*.pdf"
        
        import glob
        pdf_files = glob.glob(str(pdf_pattern))
        if not pdf_files:
            return None
        
        # Return most recent
        pdf_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return pdf_files[0]
    
    def run_fabric_inspector(self):
        """Run the fabric inspector in a separate terminal and monitor for reports"""
        print("🚀 Starting Fabric Inspector in new terminal...")
        print("📝 Monitoring for PDF reports...")
        print("-" * 60)
        
        try:
            # Get absolute path to fabric inspector
            fabric_inspector_abs = os.path.abspath(self.fabric_inspector_path)
            python_abs = sys.executable
            
            # Launch fabric inspector in a new terminal 
            # Keep terminal visible for OpenCV windows to display properly
            terminal_commands = [
                ['gnome-terminal', '--geometry=80x24+0+0', '--', python_abs, fabric_inspector_abs],
                ['xterm', '-geometry', '80x24+0+0', '-e', python_abs, fabric_inspector_abs],
                ['konsole', '--hide-menubar', '--hide-tabbar', '-e', python_abs, fabric_inspector_abs],
                ['x-terminal-emulator', '-e', python_abs, fabric_inspector_abs],
            ]
            
            process_launched = False
            for cmd in terminal_commands:
                try:
                    subprocess.Popen(cmd, cwd=os.getcwd())
                    print(f"✅ Fabric Inspector launched in {cmd[0]}")
                    process_launched = True
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"❌ Failed to launch with {cmd[0]}: {e}")
                    continue
            
            if not process_launched:
                print("❌ Could not launch fabric inspector in new terminal")
                print("💡 Available terminal emulators not found")
                return
            
            # Monitor Documents folder for new PDF reports
            print("👀 Monitoring for new PDF reports...")
            initial_reports = set()
            documents = Path.home() / "Documents"
            
            # Get initial PDF list
            import glob
            initial_pdfs = glob.glob(str(documents / "GLASS_Report_*.pdf"))
            initial_reports.update(initial_pdfs)
            
            # Monitor for new reports
            while True:
                time.sleep(2)  # Check every 2 seconds
                
                current_pdfs = glob.glob(str(documents / "GLASS_Report_*.pdf"))
                current_reports = set(current_pdfs)
                
                # Check for new reports
                new_reports = current_reports - initial_reports
                
                if new_reports:
                    for new_report in new_reports:
                        print(f"📊 New report detected: {os.path.basename(new_report)}")
                        print("🚀 Opening PDF...")
                        self.open_pdf_report(new_report)
                        self.reports_generated.append(new_report)
                        initial_reports.add(new_report)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error monitoring reports: {e}")
    
    def run(self):
        """Main orchestrator - simply run fabric inspector and handle reports"""
        self.print_banner()
        
        if not self.check_fabric_inspector():
            return
        
        print("✅ Fabric Inspector found")
        print("🎯 Starting fabric inspection workflow...")
        
        # Just run the fabric inspector directly
        self.run_fabric_inspector()
        
        print("\n🏁 Orchestrator completed")

def main():
    """Main entry point"""
    orchestrator = FabricOrchestrator()
    orchestrator.run()

if __name__ == '__main__':
    main()
