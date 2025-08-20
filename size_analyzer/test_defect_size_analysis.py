#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from size_analyzer import DefectSizeAnalyzer, DefectMetrics
import json

def create_synthetic_anomaly_mask(width=384, height=384):
    """Create a synthetic anomaly mask for testing"""
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Add several defects of different sizes
    # Large defect (ellipse)
    cv2.ellipse(mask, (100, 100), (30, 20), 0, 0, 360, 0.9, -1)
    
    # Medium defects (circles)
    cv2.circle(mask, (200, 150), 15, 0.7, -1)
    cv2.circle(mask, (300, 80), 12, 0.8, -1)
    
    # Small defects
    cv2.circle(mask, (150, 250), 8, 0.6, -1)
    cv2.circle(mask, (250, 280), 6, 0.5, -1)
    cv2.circle(mask, (320, 200), 10, 0.75, -1)
    
    # Very small defects (noise-like)
    for i in range(10):
        x, y = np.random.randint(50, width-50, 2)
        cv2.circle(mask, (x, y), np.random.randint(3, 6), 0.4, -1)
    
    # Add some Gaussian noise to simulate real GLASS output
    noise = np.random.normal(0, 0.05, mask.shape)
    mask = np.clip(mask + noise, 0, 1)
    
    return mask

def test_defect_size_analysis():
    """Test the defect size analysis functionality"""
    print("Testing GLASS Defect Size Analysis")
    print("=" * 50)
    
    # Create synthetic data
    width, height = 384, 384
    synthetic_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    synthetic_mask = create_synthetic_anomaly_mask(width, height)
    
    # Test different analyzer configurations
    analyzers = [
        ("Basic Analysis", DefectSizeAnalyzer()),
        ("With Physical Measurements", DefectSizeAnalyzer(pixel_size=0.1, physical_unit="mm")),
        ("High Sensitivity", DefectSizeAnalyzer(min_defect_area=5)),
    ]
    
    thresholds = [0.3, 0.5, 0.7]
    
    for analyzer_name, analyzer in analyzers:
        print(f"\\n{analyzer_name}:")
        print("-" * len(analyzer_name))
        
        for threshold in thresholds:
            print(f"\\nThreshold: {threshold}")
            
            # Analyze defects
            metrics = analyzer.analyze_defects(synthetic_mask, threshold=threshold)
            
            # Print results
            print(f"  Total defects: {metrics.num_defects}")
            print(f"  Total defect area: {metrics.total_defect_pixels} pixels ({metrics.defect_percentage:.2f}%)")
            
            if metrics.num_defects > 0:
                print(f"  Largest defect: {metrics.largest_defect_area} pixels")
                print(f"  Average defect size: {metrics.average_defect_area:.1f} pixels")
                print(f"  Median defect size: {metrics.median_defect_area:.1f} pixels")
            
            # Physical measurements
            if metrics.physical_unit:
                print(f"  Physical total area: {metrics.total_defect_area_physical:.3f} {metrics.physical_unit}²")
                if metrics.num_defects > 0:
                    avg_phys_area = np.mean(metrics.defect_areas_physical) if metrics.defect_areas_physical else 0
                    print(f"  Average physical size: {avg_phys_area:.3f} {metrics.physical_unit}²")
            
            # Severity levels
            print("  Severity levels:")
            for level, count in metrics.severity_levels.items():
                print(f"    {level}: {count} pixels")
    
    # Test visualization
    print(f"\\n\\nCreating visualizations...")
    
    # Use the analyzer with physical measurements for visualization
    analyzer = DefectSizeAnalyzer(pixel_size=0.1, physical_unit="mm", min_defect_area=10)
    metrics = analyzer.analyze_defects(synthetic_mask, threshold=0.5)
    
    # Create defect visualization
    vis_image = analyzer.create_defect_visualization(synthetic_image, synthetic_mask, metrics, threshold=0.5)
    
    # Save visualization
    cv2.imwrite('test_defect_visualization.png', vis_image)
    print("Saved defect visualization: test_defect_visualization.png")
    
    # Export measurements
    analyzer.export_measurements(metrics, 'test_defect_measurements.json')
    print("Saved measurements: test_defect_measurements.json")
    
    # Create size distribution plot
    if metrics.num_defects > 0:
        analyzer.create_size_distribution_plot(metrics, 'test_size_distribution.png')
        print("Saved size distribution plot: test_size_distribution.png")
    
    # Display summary
    print(f"\\n\\nFINAL SUMMARY:")
    print(f"Analyzed {width}x{height} synthetic image")
    print(f"Found {metrics.num_defects} defects at threshold 0.5")
    print(f"Total defect coverage: {metrics.defect_percentage:.2f}%")
    
    if metrics.physical_unit:
        print(f"Total physical defect area: {metrics.total_defect_area_physical:.3f} {metrics.physical_unit}²")
    
    return metrics

def test_batch_analysis():
    """Test batch analysis functionality"""
    print("\\n\\nTesting Batch Analysis")
    print("=" * 30)
    
    # Create multiple synthetic frames
    frames_data = []
    for i in range(5):
        # Create image and mask with varying defect levels
        image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        mask = create_synthetic_anomaly_mask()
        
        # Vary defect intensity
        mask = mask * (0.5 + 0.5 * np.random.random())
        
        frames_data.append((image, mask))
    
    # Analyze batch
    from size_analyzer.defect_size_analyzer import batch_analyze_video_frames
    results = batch_analyze_video_frames(frames_data, pixel_size=0.1, threshold=0.5)
    
    # Print batch results
    print(f"Analyzed {len(results)} frames:")
    for i, metrics in enumerate(results):
        print(f"Frame {i+1}: {metrics.num_defects} defects, {metrics.defect_percentage:.2f}% coverage")
    
    # Calculate batch statistics
    total_defects = sum(m.num_defects for m in results)
    avg_defects_per_frame = total_defects / len(results)
    avg_coverage = np.mean([m.defect_percentage for m in results])
    
    print(f"\\nBatch Statistics:")
    print(f"Total defects across all frames: {total_defects}")
    print(f"Average defects per frame: {avg_defects_per_frame:.2f}")
    print(f"Average coverage per frame: {avg_coverage:.2f}%")

if __name__ == '__main__':
    # Run tests
    try:
        metrics = test_defect_size_analysis()
        test_batch_analysis()
        
        print("\\n" + "=" * 50)
        print("All tests completed successfully!")
        print("Check the generated files:")
        print("- test_defect_visualization.png")
        print("- test_defect_measurements.json") 
        print("- test_size_distribution.png")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()