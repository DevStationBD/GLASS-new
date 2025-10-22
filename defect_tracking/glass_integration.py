#!/usr/bin/env python3
"""
GLASS Integration Layer for Defect Tracking

This module provides seamless integration between GLASS anomaly detection
and the FabricByteTracker for temporal defect tracking.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass

# Import GLASS components
try:
    from size_analyzer import DefectSizeAnalyzer, DefectMetrics
except ImportError:
    # Fallback for development/testing
    DefectSizeAnalyzer = None
    DefectMetrics = None

from .fabric_bytetrack import FabricByteTracker, DefectDetection, DefectTrack

logger = logging.getLogger(__name__)

@dataclass
class TrackedDefectSummary:
    """Summary of a tracked defect's lifecycle"""
    track_id: int
    defect_type: str
    first_frame: int
    last_frame: int
    duration_frames: int
    max_confidence: float
    avg_confidence: float
    max_area_pixels: float
    avg_area_pixels: float
    max_area_physical: float  # In mm² or specified unit
    avg_area_physical: float
    fabric_start_position: float
    fabric_end_position: float
    fabric_length_affected: float
    centroid_trajectory: List[Tuple[float, float]]


class GLASSDefectTracker:
    """Integration layer between GLASS detection and ByteTrack tracking"""
    
    def __init__(self, 
                 pixel_size_mm: float = 0.1,
                 high_conf_threshold: float = 0.7,
                 low_conf_threshold: float = 0.3,
                 min_defect_area: int = 10,
                 detection_threshold: float = 0.5,
                 initial_fabric_speed: float = 5.0):
        """
        Initialize GLASS defect tracker with adaptive fabric speed estimation
        
        Args:
            pixel_size_mm: Physical size of one pixel in mm
            high_conf_threshold: High confidence threshold for tracking
            low_conf_threshold: Low confidence threshold for tracking
            min_defect_area: Minimum defect area in pixels to consider
            detection_threshold: GLASS anomaly threshold for detection
            initial_fabric_speed: Arbitrary starting speed (auto-adapts to real speed)
        """
        # Initialize size analyzer if available
        if DefectSizeAnalyzer is not None:
            self.size_analyzer = DefectSizeAnalyzer(
                pixel_size=pixel_size_mm,
                physical_unit="mm",
                min_defect_area=min_defect_area
            )
        else:
            self.size_analyzer = None
            logger.warning("DefectSizeAnalyzer not available, using fallback detection")
        
        # Initialize ByteTrack tracker with adaptive speed
        self.tracker = FabricByteTracker(
            initial_fabric_speed=initial_fabric_speed,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold
        )
        
        self.pixel_size_mm = pixel_size_mm
        self.detection_threshold = detection_threshold
        self.frame_count = 0
        
        # Tracking state
        self.all_tracks: List[DefectTrack] = []
        self.completed_tracks: List[DefectTrack] = []
        
        logger.info(f"GLASSDefectTracker initialized with pixel_size={pixel_size_mm}mm")
        
    def process_frame(self, 
                     frame: np.ndarray, 
                     anomaly_mask: np.ndarray, 
                     fabric_motion: Tuple[float, float] = (0, 0),
                     debug: bool = False) -> List[DefectTrack]:
        """
        Process single frame: GLASS detection → ByteTrack tracking
        
        Args:
            frame: Original frame (H, W, 3)
            anomaly_mask: GLASS anomaly probability mask (H, W) with values 0-1
            fabric_motion: Estimated fabric movement (dx, dy) since last frame
            debug: Enable debug visualizations
            
        Returns:
            List of active defect tracks
        """
        self.frame_count += 1
        
        # Extract defects using GLASS size analyzer or fallback method
        if self.size_analyzer is not None:
            detections = self._extract_defects_with_analyzer(anomaly_mask)
        else:
            detections = self._extract_defects_fallback(anomaly_mask)
        
        logger.debug(f"Frame {self.frame_count}: Extracted {len(detections)} defects")
        
        # Update tracker with new detections
        active_tracks = self.tracker.update(detections, fabric_motion)
        
        # Store all tracks for later analysis
        for track in active_tracks:
            if track.track_id not in [t.track_id for t in self.all_tracks]:
                self.all_tracks.append(track)
        
        # Check for completed tracks (tracks that haven't been seen recently)
        self._update_completed_tracks()
        
        if debug:
            self._log_frame_statistics(detections, active_tracks)
        
        return active_tracks
    
    def _extract_defects_with_analyzer(self, anomaly_mask: np.ndarray) -> List[DefectDetection]:
        """Extract defects using GLASS DefectSizeAnalyzer"""
        try:
            # Analyze defects with size analyzer
            defect_metrics = self.size_analyzer.analyze_defects(
                anomaly_mask, 
                threshold=self.detection_threshold,
                use_morphology=True
            )
            
            detections = []
            for i in range(defect_metrics.num_defects):
                bbox = defect_metrics.defect_bounding_boxes[i]
                centroid = defect_metrics.defect_centroids[i]
                area = defect_metrics.defect_areas[i]
                
                # Calculate confidence from anomaly mask region
                x, y, w, h = [int(v) for v in bbox]
                roi = anomaly_mask[y:y+h, x:x+w]
                confidence = float(np.mean(roi)) if roi.size > 0 else 0.5
                
                detection = DefectDetection(
                    bbox=bbox,
                    confidence=confidence,
                    centroid=centroid,
                    area=area,
                    frame_id=self.frame_count,
                    defect_type=self._classify_defect_type(roi, area)
                )
                detections.append(detection)
                
            return detections
            
        except Exception as e:
            logger.error(f"Error in defect extraction: {e}")
            return self._extract_defects_fallback(anomaly_mask)
    
    def _extract_defects_fallback(self, anomaly_mask: np.ndarray) -> List[DefectDetection]:
        """Fallback defect extraction using basic computer vision"""
        # Ensure mask is in correct format
        if anomaly_mask.max() <= 1.0:
            binary_mask = (anomaly_mask >= self.detection_threshold).astype(np.uint8) * 255
        else:
            binary_mask = (anomaly_mask >= self.detection_threshold * 255).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < 10:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w/2, y + h/2)
            
            # Calculate confidence from mask region
            roi = anomaly_mask[y:y+h, x:x+w]
            confidence = float(np.mean(roi)) if roi.size > 0 else 0.5
            
            detection = DefectDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                centroid=centroid,
                area=area,
                frame_id=self.frame_count,
                defect_type="unknown"
            )
            detections.append(detection)
        
        return detections
    
    def _classify_defect_type(self, roi: np.ndarray, area: float) -> str:
        """Simple defect type classification based on size and shape"""
        if area < 50:
            return "spot"
        elif area < 200:
            return "small_defect"
        elif area < 500:
            return "medium_defect"
        else:
            return "large_defect"
    
    def _update_completed_tracks(self):
        """Move tracks that haven't been seen recently to completed list"""
        current_active_ids = {t.track_id for t in self.tracker.tracks if t.state == 'active'}
        
        for track in self.all_tracks:
            if (track.track_id not in current_active_ids and 
                track.track_id not in [t.track_id for t in self.completed_tracks] and
                len(track.detections) > 0):
                
                # Mark track as completed if it hasn't been active for a while
                if self.frame_count - track.last_frame > self.tracker.max_lost_frames:
                    self.completed_tracks.append(track)
                    logger.debug(f"Track {track.track_id} completed after {track.duration_frames} frames")
    
    def _log_frame_statistics(self, detections: List[DefectDetection], active_tracks: List[DefectTrack]):
        """Log frame processing statistics"""
        logger.debug(f"Frame {self.frame_count} statistics:")
        logger.debug(f"  - Detections: {len(detections)}")
        logger.debug(f"  - Active tracks: {len(active_tracks)}")
        logger.debug(f"  - Total tracks created: {self.tracker.total_tracks_created}")
        logger.debug(f"  - Completed tracks: {len(self.completed_tracks)}")
    
    def _merge_fragmented_tracks(self, tracks: List[DefectTrack]) -> List[DefectTrack]:
        """Merge tracks that are likely fragments of the same physical defect"""
        if len(tracks) <= 1:
            return tracks.copy()
        
        # Sort tracks by first frame
        sorted_tracks = sorted(tracks, key=lambda t: t.first_frame)
        merged_tracks = []
        used_track_ids = set()
        
        for i, track1 in enumerate(sorted_tracks):
            if track1.track_id in used_track_ids:
                continue
                
            # Start with current track
            merged_track = track1
            used_track_ids.add(track1.track_id)
            
            # Look for nearby tracks that could be the same defect
            for j, track2 in enumerate(sorted_tracks[i+1:], i+1):
                if track2.track_id in used_track_ids:
                    continue
                
                # Check if tracks are close in time and space
                if self._should_merge_tracks(track1, track2):
                    # Merge track2 into track1
                    merged_track = self._merge_two_tracks(merged_track, track2)
                    used_track_ids.add(track2.track_id)
                    logger.debug(f"Merged track {track2.track_id} into {track1.track_id}")
            
            merged_tracks.append(merged_track)
        
        logger.info(f"Track merging: {len(tracks)} → {len(merged_tracks)} defects")
        return merged_tracks
    
    def _should_merge_tracks(self, track1: DefectTrack, track2: DefectTrack) -> bool:
        """Determine if two tracks should be merged as the same physical defect"""
        # Only merge very short tracks (likely fragmentation)
        if track1.duration_frames > 5 and track2.duration_frames > 5:
            return False
        
        # Check temporal proximity (within 20 frames)
        time_gap = abs(track1.first_frame - track2.last_frame)
        if time_gap > 20:
            return False
        
        # Check spatial proximity at fabric level
        pos1 = track1.fabric_start_position
        pos2 = track2.fabric_start_position
        
        # Defects should be close in fabric position (~50 pixels considering fabric movement)
        spatial_distance = abs(pos1 - pos2)
        fabric_distance_threshold = 50.0  # pixels
        
        return spatial_distance < fabric_distance_threshold
    
    def _merge_two_tracks(self, track1: DefectTrack, track2: DefectTrack) -> DefectTrack:
        """Merge two tracks into one, keeping track1 as base"""
        # Create new merged track with track1 as base
        merged_track = DefectTrack(
            track_id=track1.track_id,  # Keep original ID
            state=track1.state,
            age=track1.age,
            fabric_start_position=min(track1.fabric_start_position, track2.fabric_start_position),
            kalman_filter=track1.kalman_filter,
            last_prediction=track1.last_prediction
        )
        
        # Combine all detections and sort by frame
        all_detections = track1.detections + track2.detections
        all_detections.sort(key=lambda d: d.frame_id)
        
        # Add all detections to merged track
        for detection in all_detections:
            merged_track.add_detection(detection)
        
        return merged_track
    
    def get_tracked_defect_summaries(self) -> List[TrackedDefectSummary]:
        """Get summary of all tracked defects"""
        summaries = []
        
        # Use only all_tracks since completed tracks are already included
        # Apply post-processing to merge fragmented tracks that are likely the same defect
        merged_tracks = self._merge_fragmented_tracks(self.all_tracks)
        
        for track in merged_tracks:
            if len(track.detections) == 0:
                continue
            
            # Calculate trajectory
            trajectory = [d.centroid for d in track.detections]
            
            # Calculate fabric positions
            start_pos = track.fabric_start_position
            end_pos = track.detections[-1].centroid[0] if track.detections else start_pos
            length_affected = abs(end_pos - start_pos) * self.pixel_size_mm
            
            # Convert areas to physical units
            max_area_physical = track.max_area * (self.pixel_size_mm ** 2)
            avg_area_physical = track.avg_area * (self.pixel_size_mm ** 2)
            
            summary = TrackedDefectSummary(
                track_id=track.track_id,
                defect_type=track.detections[0].defect_type,
                first_frame=track.first_frame,
                last_frame=track.last_frame,
                duration_frames=track.duration_frames,
                max_confidence=track.max_confidence,
                avg_confidence=track.avg_confidence,
                max_area_pixels=track.max_area,
                avg_area_pixels=track.avg_area,
                max_area_physical=max_area_physical,
                avg_area_physical=avg_area_physical,
                fabric_start_position=start_pos * self.pixel_size_mm,
                fabric_end_position=end_pos * self.pixel_size_mm,
                fabric_length_affected=length_affected,
                centroid_trajectory=trajectory
            )
            summaries.append(summary)
        
        return summaries
    
    def get_tracking_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        base_stats = self.tracker.get_statistics()
        
        # Add GLASS-specific statistics
        summaries = self.get_tracked_defect_summaries()
        
        defect_types = {}
        total_physical_area = 0.0
        
        for summary in summaries:
            # Count by defect type
            defect_types[summary.defect_type] = defect_types.get(summary.defect_type, 0) + 1
            total_physical_area += summary.max_area_physical
        
        enhanced_stats = {
            **base_stats,
            'unique_defects_tracked': len(summaries),
            'completed_tracks': len(self.completed_tracks),
            'defect_types_distribution': defect_types,
            'total_defect_area_mm2': total_physical_area,
            'avg_defect_duration': np.mean([s.duration_frames for s in summaries]) if summaries else 0,
            'max_defect_duration': max([s.duration_frames for s in summaries], default=0)
        }
        
        return enhanced_stats
    
    def visualize_tracks_on_frame(self, frame: np.ndarray, active_tracks: List[DefectTrack]) -> np.ndarray:
        """Visualize active tracks on frame"""
        vis_frame = frame.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, track in enumerate(active_tracks):
            if not track.detections:
                continue
                
            color = colors[i % len(colors)]
            last_detection = track.detections[-1]
            
            # Draw bounding box
            x, y, w, h = [int(v) for v in last_detection.bbox]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track.track_id} ({last_detection.confidence:.2f})"
            cv2.putText(vis_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw trajectory if track has multiple detections
            if len(track.detections) > 1:
                for j in range(1, len(track.detections)):
                    pt1 = tuple(map(int, track.detections[j-1].centroid))
                    pt2 = tuple(map(int, track.detections[j].centroid))
                    cv2.line(vis_frame, pt1, pt2, color, 1)
        
        return vis_frame
    
    def reset(self):
        """Reset tracker state"""
        self.tracker = FabricByteTracker(
            initial_fabric_speed=self.tracker.speed_tracker.get_current_speed(),
            high_conf_threshold=self.tracker.base_high_conf_thresh,
            low_conf_threshold=self.tracker.base_low_conf_thresh
        )
        self.frame_count = 0
        self.all_tracks = []
        self.completed_tracks = []
        logger.info("GLASSDefectTracker reset")