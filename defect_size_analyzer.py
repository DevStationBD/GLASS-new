#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
import matplotlib.pyplot as plt
import json

@dataclass
class DefectMetrics:
    """Container for defect measurement results"""
    # Overall metrics
    total_defect_pixels: int
    total_image_pixels: int
    defect_percentage: float
    
    # Individual defect metrics
    num_defects: int
    defect_areas: List[int]  # Areas of individual defects in pixels
    defect_centroids: List[Tuple[float, float]]  # (x, y) coordinates
    defect_bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    
    # Size statistics
    largest_defect_area: int
    average_defect_area: float
    median_defect_area: float
    
    # Multi-threshold analysis
    severity_levels: Dict[str, int]  # pixel counts at different thresholds
    
    # Physical measurements (if pixel_size provided)
    physical_unit: Optional[str] = None
    total_defect_area_physical: Optional[float] = None
    defect_areas_physical: Optional[List[float]] = None

class DefectSizeAnalyzer:
    """Comprehensive defect size measurement from GLASS anomaly masks"""
    
    def __init__(self, 
                 pixel_size: Optional[float] = None, 
                 physical_unit: str = "mm",
                 min_defect_area: int = 10):
        """
        Initialize defect size analyzer
        
        Args:
            pixel_size: Size of one pixel in physical units (e.g., 0.1 for 0.1mm per pixel)
            physical_unit: Unit for physical measurements (mm, cm, inch, etc.)
            min_defect_area: Minimum defect size in pixels to consider
        """
        self.pixel_size = pixel_size
        self.physical_unit = physical_unit
        self.min_defect_area = min_defect_area
        
        # Severity thresholds for multi-level analysis
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    
    def analyze_defects(self, 
                       anomaly_mask: np.ndarray, 
                       threshold: float = 0.5,
                       use_morphology: bool = True) -> DefectMetrics:
        """
        Comprehensive defect size analysis
        
        Args:
            anomaly_mask: GLASS anomaly probability mask (0-1 float values)
            threshold: Binary threshold for defect detection
            use_morphology: Apply morphological operations to clean up mask
            
        Returns:
            DefectMetrics object with comprehensive measurements
        """
        # Ensure mask is float and normalized
        mask = anomaly_mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Create binary mask
        binary_mask = (mask >= threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up
        if use_morphology:
            # Remove small noise
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_noise)
            
            # Fill small holes
            kernel_holes = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_holes)
        
        # Overall metrics
        total_defect_pixels = np.sum(binary_mask)
        total_image_pixels = binary_mask.size
        defect_percentage = (total_defect_pixels / total_image_pixels) * 100
        
        # Connected component analysis for individual defects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8)
        
        # Filter out background (label 0) and small components
        valid_components = []
        defect_areas = []
        defect_centroids = []
        defect_bounding_boxes = []
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_defect_area:
                valid_components.append(i)
                defect_areas.append(area)
                defect_centroids.append((centroids[i][0], centroids[i][1]))
                
                # Bounding box: (x, y, width, height)
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                defect_bounding_boxes.append((x, y, w, h))
        
        # Size statistics
        num_defects = len(defect_areas)
        largest_defect_area = max(defect_areas) if defect_areas else 0
        average_defect_area = np.mean(defect_areas) if defect_areas else 0
        median_defect_area = np.median(defect_areas) if defect_areas else 0
        
        # Multi-threshold severity analysis
        severity_levels = {}
        for level, thresh in self.severity_thresholds.items():
            severity_mask = (mask >= thresh).astype(np.uint8)
            severity_levels[level] = np.sum(severity_mask)
        
        # Physical measurements
        physical_unit = None
        total_defect_area_physical = None
        defect_areas_physical = None
        
        if self.pixel_size is not None:
            physical_unit = self.physical_unit
            pixel_area = self.pixel_size ** 2  # area per pixel
            total_defect_area_physical = total_defect_pixels * pixel_area
            defect_areas_physical = [area * pixel_area for area in defect_areas]
        
        return DefectMetrics(
            total_defect_pixels=total_defect_pixels,
            total_image_pixels=total_image_pixels,
            defect_percentage=defect_percentage,
            num_defects=num_defects,
            defect_areas=defect_areas,
            defect_centroids=defect_centroids,
            defect_bounding_boxes=defect_bounding_boxes,
            largest_defect_area=largest_defect_area,
            average_defect_area=average_defect_area,
            median_defect_area=median_defect_area,
            severity_levels=severity_levels,
            physical_unit=physical_unit,
            total_defect_area_physical=total_defect_area_physical,
            defect_areas_physical=defect_areas_physical
        )
    
    def create_defect_visualization(self, 
                                  original_image: np.ndarray,
                                  anomaly_mask: np.ndarray,
                                  metrics: DefectMetrics,
                                  threshold: float = 0.5) -> np.ndarray:
        """
        Create visualization with defect measurements
        
        Args:
            original_image: Original input image
            anomaly_mask: GLASS anomaly probability mask
            metrics: DefectMetrics from analyze_defects()
            threshold: Binary threshold used for analysis
            
        Returns:
            Annotated image with defect measurements
        """
        # Create a copy for annotation
        vis_image = original_image.copy()
        h, w = vis_image.shape[:2]
        
        # Resize mask to match image if needed
        if anomaly_mask.shape != (h, w):
            mask_resized = cv2.resize(anomaly_mask, (w, h))
        else:
            mask_resized = anomaly_mask.copy()
        
        # Create binary mask for contours
        binary_mask = (mask_resized >= threshold).astype(np.uint8)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find and draw contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw defect boundaries and measurements
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= self.min_defect_area:
                # Draw contour
                cv2.drawContours(vis_image, [contour], -1, (0, 0, 255), 2)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw centroid
                    cv2.circle(vis_image, (cx, cy), 3, (255, 0, 0), -1)
                    
                    # Add area text
                    area_text = f"{int(area)}px"
                    if self.pixel_size is not None:
                        physical_area = area * (self.pixel_size ** 2)
                        area_text += f"\n{physical_area:.2f}{self.physical_unit}²"
                    
                    # Position text above the defect
                    text_y = max(y - 10, 20)
                    cv2.putText(vis_image, f"#{i+1}", (x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(vis_image, area_text, (x, text_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add overall statistics
        stats_text = [
            f"Total Defects: {metrics.num_defects}",
            f"Total Area: {metrics.total_defect_pixels}px ({metrics.defect_percentage:.2f}%)"
        ]
        
        if metrics.physical_unit:
            stats_text.append(f"Physical Area: {metrics.total_defect_area_physical:.2f}{metrics.physical_unit}²")
        
        if metrics.num_defects > 0:
            stats_text.extend([
                f"Largest: {metrics.largest_defect_area}px",
                f"Average: {metrics.average_defect_area:.1f}px"
            ])
        
        # Draw statistics box
        box_height = len(stats_text) * 20 + 10
        cv2.rectangle(vis_image, (10, 10), (300, 10 + box_height), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (10, 10), (300, 10 + box_height), (255, 255, 255), 1)
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_image, text, (15, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def export_measurements(self, 
                          metrics: DefectMetrics, 
                          filename: str,
                          include_coordinates: bool = True) -> None:
        """
        Export defect measurements to JSON file
        
        Args:
            metrics: DefectMetrics to export
            filename: Output JSON filename
            include_coordinates: Include centroid and bounding box data
        """
        data = {
            'summary': {
                'total_defect_pixels': int(metrics.total_defect_pixels),
                'total_image_pixels': int(metrics.total_image_pixels),
                'defect_percentage': float(metrics.defect_percentage),
                'num_defects': int(metrics.num_defects),
                'largest_defect_area': int(metrics.largest_defect_area),
                'average_defect_area': float(metrics.average_defect_area),
                'median_defect_area': float(metrics.median_defect_area)
            },
            'severity_levels': {k: int(v) for k, v in metrics.severity_levels.items()},
            'individual_defects': []
        }
        
        # Add physical measurements if available
        if metrics.physical_unit:
            data['physical_measurements'] = {
                'unit': metrics.physical_unit,
                'total_defect_area': metrics.total_defect_area_physical,
                'defect_areas': metrics.defect_areas_physical
            }
        
        # Add individual defect data
        for i in range(metrics.num_defects):
            defect_data = {
                'id': i + 1,
                'area_pixels': int(metrics.defect_areas[i])
            }
            
            if metrics.defect_areas_physical:
                defect_data['area_physical'] = float(metrics.defect_areas_physical[i])
            
            if include_coordinates:
                defect_data['centroid'] = {
                    'x': float(metrics.defect_centroids[i][0]),
                    'y': float(metrics.defect_centroids[i][1])
                }
                defect_data['bounding_box'] = {
                    'x': int(metrics.defect_bounding_boxes[i][0]),
                    'y': int(metrics.defect_bounding_boxes[i][1]),
                    'width': int(metrics.defect_bounding_boxes[i][2]),
                    'height': int(metrics.defect_bounding_boxes[i][3])
                }
            
            data['individual_defects'].append(defect_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_size_distribution_plot(self, 
                                    metrics: DefectMetrics, 
                                    save_path: Optional[str] = None) -> None:
        """
        Create histogram of defect size distribution
        
        Args:
            metrics: DefectMetrics to plot
            save_path: Optional path to save plot image
        """
        if metrics.num_defects == 0:
            print("No defects to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        areas = metrics.defect_areas_physical if metrics.defect_areas_physical else metrics.defect_areas
        unit = f"{metrics.physical_unit}²" if metrics.physical_unit else "pixels"
        
        plt.hist(areas, bins=min(20, len(areas)), alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.1f}')
        plt.axvline(np.median(areas), color='green', linestyle='--', label=f'Median: {np.median(areas):.1f}')
        
        plt.xlabel(f'Defect Area ({unit})')
        plt.ylabel('Count')
        plt.title(f'Defect Size Distribution (n={metrics.num_defects})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage functions
def analyze_single_image(image_path: str, 
                        model_mask: np.ndarray, 
                        pixel_size: Optional[float] = None,
                        threshold: float = 0.5) -> DefectMetrics:
    """
    Analyze defects in a single image
    
    Args:
        image_path: Path to original image
        model_mask: GLASS anomaly probability mask
        pixel_size: Physical size per pixel (optional)
        threshold: Detection threshold
        
    Returns:
        DefectMetrics with measurements
    """
    analyzer = DefectSizeAnalyzer(pixel_size=pixel_size)
    metrics = analyzer.analyze_defects(model_mask, threshold=threshold)
    return metrics

def batch_analyze_video_frames(video_results: List[Tuple[np.ndarray, np.ndarray]], 
                              pixel_size: Optional[float] = None,
                              threshold: float = 0.5) -> List[DefectMetrics]:
    """
    Batch analyze defects in video frames
    
    Args:
        video_results: List of (frame, mask) tuples
        pixel_size: Physical size per pixel (optional)  
        threshold: Detection threshold
        
    Returns:
        List of DefectMetrics for each frame
    """
    analyzer = DefectSizeAnalyzer(pixel_size=pixel_size)
    results = []
    
    for frame, mask in video_results:
        metrics = analyzer.analyze_defects(mask, threshold=threshold)
        results.append(metrics)
    
    return results