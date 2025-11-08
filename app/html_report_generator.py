#!/usr/bin/env python3
"""
HTML Report Generator for GLASS Inference Results

Creates HTML reports that can be opened in any web browser as a fallback
when PDF generation is not available.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import webbrowser

class GLASSHTMLReportGenerator:
    """Generate HTML reports from GLASS inference results"""
    
    def __init__(self):
        # Create HTML template with proper escaping
        self.create_html_template()
    
    def create_html_template(self):
        """Create HTML template with proper formatting"""
        self.html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLASS Fabric Inspection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 3px solid #2c5aa0; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ color: #2c5aa0; margin: 0; font-size: 2.5em; }}
        .header .subtitle {{ color: #666; font-size: 1.1em; margin-top: 10px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c5aa0; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .info-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #2c5aa0; }}
        .info-card h3 {{ margin: 0 0 15px 0; color: #2c5aa0; }}
        .info-item {{ display: flex; justify-content: space-between; margin-bottom: 8px; padding: 5px 0; border-bottom: 1px solid #e0e0e0; }}
        .info-item:last-child {{ border-bottom: none; }}
        .info-label {{ font-weight: 600; color: #555; }}
        .info-value {{ color: #333; font-family: monospace; }}
        .defects-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .defects-table th {{ background: #2c5aa0; color: white; padding: 15px 10px; text-align: center; }}
        .defects-table td {{ padding: 12px 10px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
        .defects-table tr:nth-child(even) {{ background: #f8f9fa; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .no-defects {{ text-align: center; padding: 40px; color: #28a745; font-size: 1.2em; background: #d4edda; border-radius: 8px; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        {{content}}
    </div>
</body>
</html>'''
    
    def load_report_data(self, json_path: str) -> dict:
        """Load tracking report from JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Report file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp_str
    
    def generate_html_content(self, data: dict) -> str:
        """Generate HTML content from tracking data"""
        content = []
        
        # Header
        content.append(f"""
        <div class="header">
            <h1>🔍 GLASS Fabric Inspection Report</h1>
            <div class="subtitle">Session: {data.get('session_id', 'N/A')}</div>
            <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        """)
        
        # Key Statistics
        unique_defects = data.get('unique_defects', 0)
        fps = data.get('fps_processing', 0)
        frames = data.get('frames_processed', 0)
        processing_time = data.get('processing_time_seconds', 0)
        
        content.append(f"""
        <div class="section">
            <h2>📊 Key Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">{unique_defects}</div>
                    <div class="stat-label">Unique Defects</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{fps:.1f}</div>
                    <div class="stat-label">Processing FPS</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{frames:,}</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{processing_time:.1f}s</div>
                    <div class="stat-label">Processing Time</div>
                </div>
            </div>
        </div>
        """)
        
        # Session Information
        content.append(f"""
        <div class="section">
            <div class="info-grid">
                <div class="info-card">
                    <h3>📋 Session Information</h3>
                    <div class="info-item">
                        <span class="info-label">Session ID:</span>
                        <span class="info-value">{data.get('session_id', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Timestamp:</span>
                        <span class="info-value">{self.format_timestamp(data.get('timestamp', 'N/A'))}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Video File:</span>
                        <span class="info-value">{data.get('video_file', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Output File:</span>
                        <span class="info-value">{data.get('output_file', 'N/A')}</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>⚡ Processing Performance</h3>
                    <div class="info-item">
                        <span class="info-label">Frames Processed:</span>
                        <span class="info-value">{frames:,}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Processing Time:</span>
                        <span class="info-value">{processing_time:.2f} seconds</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Processing FPS:</span>
                        <span class="info-value">{fps:.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Peak Concurrent:</span>
                        <span class="info-value">{data.get('peak_concurrent_defects', 0)}</span>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Tracking Statistics
        tracking_stats = data.get('tracking_statistics', {})
        if tracking_stats:
            content.append(f"""
            <div class="section">
                <h2>🎯 Tracking Analysis</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>🔍 Detection Statistics</h3>
                        <div class="info-item">
                            <span class="info-label">Total Detections:</span>
                            <span class="info-value">{tracking_stats.get('total_detections', 0):,}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Avg per Frame:</span>
                            <span class="info-value">{tracking_stats.get('avg_detections_per_frame', 0):.2f}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Tracks Created:</span>
                            <span class="info-value">{tracking_stats.get('total_tracks_created', 0)}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Completed Tracks:</span>
                            <span class="info-value">{tracking_stats.get('completed_tracks', 0)}</span>
                        </div>
                    </div>
                    
                    <div class="info-card">
                        <h3>🏃 Fabric Motion</h3>
                        <div class="info-item">
                            <span class="info-label">Current Speed:</span>
                            <span class="info-value">{tracking_stats.get('current_fabric_speed', 0):.2f} mm/s</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Fabric State:</span>
                            <span class="info-value">{tracking_stats.get('fabric_state', 'unknown')}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Speed Confidence:</span>
                            <span class="info-value">{tracking_stats.get('speed_confidence', 0):.3f}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Bootstrapped:</span>
                            <span class="info-value">{'Yes' if tracking_stats.get('is_speed_bootstrapped', False) else 'No'}</span>
                        </div>
                    </div>
                </div>
            </div>
            """)
        
        # Defects Table
        tracked_defects = data.get('tracked_defects', [])
        content.append('<div class="section"><h2>🎭 Detected Defects</h2>')
        
        if not tracked_defects:
            content.append('<div class="no-defects">✅ No defects detected in this session</div>')
        else:
            content.append(f"""
            <div class="highlight">
                <strong>Found {len(tracked_defects)} defect(s)</strong> during inspection
            </div>
            <table class="defects-table">
                <thead>
                    <tr>
                        <th>Track ID</th>
                        <th>Defect Type</th>
                        <th>First Frame</th>
                        <th>Last Frame</th>
                        <th>Duration (frames)</th>
                        <th>Max Confidence</th>
                        <th>Max Area (mm²)</th>
                        <th>Trajectory Length</th>
                    </tr>
                </thead>
                <tbody>
            """)
            
            for defect in tracked_defects:
                content.append(f"""
                    <tr>
                        <td>{defect.get('track_id', '')}</td>
                        <td>{defect.get('defect_type', '')}</td>
                        <td>{defect.get('first_frame', '')}</td>
                        <td>{defect.get('last_frame', '')}</td>
                        <td>{defect.get('duration_frames', '')}</td>
                        <td>{defect.get('max_confidence', 0):.3f}</td>
                        <td>{defect.get('max_area_mm2', 0):.2f}</td>
                        <td>{defect.get('trajectory_length', '')}</td>
                    </tr>
                """)
            
            content.append('</tbody></table>')
        
        content.append('</div>')
        
        # Footer
        content.append(f"""
        <div class="footer">
            Report generated by GLASS Fabric Inspector on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <em>Automated fabric defect detection and tracking system</em>
        </div>
        """)
        
        return ''.join(content)
    
    def get_documents_folder(self) -> str:
        """Get the user's Documents folder path"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            return os.path.join(os.path.expanduser("~"), "Documents")
        elif system == "Darwin":  # macOS
            return os.path.join(os.path.expanduser("~"), "Documents")
        else:  # Linux and others
            return os.path.join(os.path.expanduser("~"), "Documents")
    
    def safe_open_html(self, html_path: str):
        """Safely open HTML file in browser to avoid Qt conflicts"""
        import threading
        import time
        import subprocess
        import platform
        
        def safe_browser_open():
            """Open browser in separate thread to avoid GUI conflicts"""
            try:
                time.sleep(0.5)  # Small delay to avoid conflicts
                
                system = platform.system()
                if system == "Linux":
                    # Use xdg-open which is more reliable on Linux
                    subprocess.Popen(['xdg-open', html_path], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    print(f"📖 HTML report opened in default browser")
                elif system == "Windows":
                    subprocess.Popen(['start', html_path], shell=True,
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    print(f"📖 HTML report opened in default browser")
                elif system == "Darwin":  # macOS
                    subprocess.Popen(['open', html_path],
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    print(f"📖 HTML report opened in default browser")
                else:
                    # Fallback to webbrowser
                    file_url = f"file://{os.path.abspath(html_path)}"
                    webbrowser.open_new_tab(file_url)
                    print(f"📖 HTML report opened in web browser")
                    
            except Exception as e:
                print(f"Could not open HTML report automatically: {e}")
                print(f"📁 HTML saved at: {html_path}")
                print(f"💡 Please manually open the file to view the report")
        
        # Run in separate thread to avoid blocking
        browser_thread = threading.Thread(target=safe_browser_open, daemon=True)
        browser_thread.start()
    
    def generate_report(self, json_path: str, output_path: str = None, open_after: bool = True) -> str:
        """Generate HTML report from JSON tracking data"""
        # Load data
        data = self.load_report_data(json_path)
        
        # Determine output path
        if output_path is None:
            documents_folder = self.get_documents_folder()
            session_id = data.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            filename = f"GLASS_Report_{session_id}.html"
            output_path = os.path.join(documents_folder, filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML content
        html_content = self.generate_html_content(data)
        full_html = self.html_template.format(content=html_content)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ HTML report generated: {output_path}")
        
        # Open in browser if requested (safe method)
        if open_after:
            self.safe_open_html(output_path)
        
        return output_path


def main():
    """Test the HTML report generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML report from GLASS tracking JSON')
    parser.add_argument('json_path', help='Path to JSON tracking report')
    parser.add_argument('--output', '-o', help='Output HTML path (optional)')
    parser.add_argument('--no-open', action='store_true', help='Do not open HTML after generation')
    
    args = parser.parse_args()
    
    generator = GLASSHTMLReportGenerator()
    
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
