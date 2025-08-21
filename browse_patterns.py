#!/usr/bin/env python3
"""
Pattern Browser for GLASS Defect Visualizations
Simple tool to browse and view generated pattern visualizations
"""

import os
import sys
from pathlib import Path
import json
import webbrowser
import tempfile

def create_pattern_browser_html(patterns_dir):
    """Create an HTML browser for pattern visualizations"""
    
    patterns_path = Path(patterns_dir)
    if not patterns_path.exists():
        print(f"❌ Patterns directory not found: {patterns_path}")
        return None
    
    # Find all class directories
    class_dirs = [d for d in patterns_path.iterdir() if d.is_dir() and d.name not in ['__pycache__']]
    
    if not class_dirs:
        print(f"❌ No class directories found in: {patterns_path}")
        return None
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GLASS Defect Pattern Browser</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .class-section {
                background: white;
                margin-bottom: 30px;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .class-title {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .stats {
                background: #ecf0f1;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
            .defect-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .defect-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                transition: transform 0.2s;
            }
            .defect-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .defect-header {
                background: #34495e;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }
            .pattern-grid-img {
                width: 100%;
                height: auto;
                cursor: pointer;
            }
            .sample-images {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                padding: 10px;
                justify-content: center;
            }
            .sample-thumb {
                width: 60px;
                height: 60px;
                object-fit: cover;
                border-radius: 4px;
                cursor: pointer;
                border: 2px solid transparent;
            }
            .sample-thumb:hover {
                border-color: #3498db;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
            }
            .modal-content {
                display: block;
                margin: auto;
                max-width: 90%;
                max-height: 90%;
                margin-top: 2%;
            }
            .close {
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover {
                color: #3498db;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 GLASS Defect Pattern Browser</h1>
            <p>Browse and inspect generated defect visualizations</p>
        </div>
    """
    
    for class_dir in sorted(class_dirs):
        stats_file = class_dir / "statistics.json"
        if not stats_file.exists():
            continue
            
        with open(stats_file) as f:
            stats = json.load(f)
        
        html_content += f"""
        <div class="class-section">
            <h2 class="class-title">📂 Class: {class_dir.name}</h2>
            <div class="stats">
                <strong>📊 Statistics:</strong> 
                {stats.get('total_samples', 0)} total samples across 
                {len(stats.get('defect_types', {}))} defect types
            </div>
            
            <div class="defect-grid">
        """
        
        # Add each defect type
        grids_dir = class_dir / "grids"
        overlays_dir = class_dir / "overlays"
        comparisons_dir = class_dir / "comparisons"
        
        for defect_type, count in stats.get('defect_types', {}).items():
            grid_file = grids_dir / f"{defect_type}_pattern_grid.png"
            
            html_content += f"""
                <div class="defect-card">
                    <div class="defect-header">
                        {defect_type.upper().replace('_', ' ')}
                        <br><small>{count} samples</small>
                    </div>
            """
            
            # Add pattern grid if exists
            if grid_file.exists():
                rel_path = grid_file.relative_to(patterns_path)
                html_content += f"""
                    <img src="{rel_path}" class="pattern-grid-img" 
                         onclick="openModal('{rel_path}')" 
                         alt="{defect_type} pattern grid">
                """
            
            # Add sample thumbnails
            overlay_type_dir = overlays_dir / defect_type
            if overlay_type_dir.exists():
                overlay_files = sorted([f for f in overlay_type_dir.glob("*.png")])[:5]
                
                if overlay_files:
                    html_content += '<div class="sample-images">'
                    for overlay_file in overlay_files:
                        rel_path = overlay_file.relative_to(patterns_path)
                        html_content += f"""
                            <img src="{rel_path}" class="sample-thumb" 
                                 onclick="openModal('{rel_path}')" 
                                 alt="{overlay_file.stem}">
                        """
                    html_content += '</div>'
            
            html_content += '</div>'
        
        html_content += """
            </div>
        </div>
        """
    
    # Add modal and JavaScript
    html_content += """
        <div id="imageModal" class="modal">
            <span class="close">&times;</span>
            <img class="modal-content" id="modalImg">
        </div>
        
        <script>
            function openModal(imageSrc) {
                var modal = document.getElementById('imageModal');
                var modalImg = document.getElementById('modalImg');
                modal.style.display = 'block';
                modalImg.src = imageSrc;
            }
            
            var modal = document.getElementById('imageModal');
            var span = document.getElementsByClassName('close')[0];
            
            span.onclick = function() {
                modal.style.display = 'none';
            }
            
            modal.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    html_file = patterns_path / "pattern_browser.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Pattern browser created: {html_file}")
    return html_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Browse GLASS defect pattern visualizations")
    parser.add_argument("--patterns_dir", default="patterns/", 
                       help="Directory containing pattern visualizations")
    parser.add_argument("--open", action="store_true", 
                       help="Open browser automatically")
    
    args = parser.parse_args()
    
    html_file = create_pattern_browser_html(args.patterns_dir)
    
    if html_file and args.open:
        print("🌐 Opening pattern browser...")
        webbrowser.open(f"file://{html_file.absolute()}")
    elif html_file:
        print(f"💡 Open this file in your browser: {html_file.absolute()}")

if __name__ == "__main__":
    main()