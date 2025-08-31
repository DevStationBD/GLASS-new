#!/usr/bin/env python3
"""
Fabric-optimized ByteTrack implementation for defect tracking

This module adapts ByteTrack for fabric defect tracking, incorporating fabric-specific
motion patterns and characteristics for improved tracking accuracy.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import logging

from .adaptive_speed import AdaptiveFabricSpeedTracker

logger = logging.getLogger(__name__)

@dataclass
class DefectDetection:
    """Single frame defect detection from GLASS"""
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float  # GLASS anomaly score (0-1)
    centroid: Tuple[float, float]  # (x, y) center point
    area: float  # Defect area in pixels
    frame_id: int  # Frame number where detected
    defect_type: str = "unknown"  # Type of defect if classified

    def __post_init__(self):
        """Validate detection data"""
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be between 0-1, got {self.confidence}")
        if self.area <= 0:
            raise ValueError(f"Area must be positive, got {self.area}")

@dataclass 
class DefectTrack:
    """Multi-frame defect track with temporal information"""
    track_id: int
    detections: List[DefectDetection] = field(default_factory=list)
    state: str = 'active'  # 'active', 'lost', 'completed'
    age: int = 0  # Frames since last detection
    fabric_start_position: float = 0.0  # Where defect first appeared on fabric
    
    # Kalman filter for motion prediction
    kalman_filter: Optional[cv2.KalmanFilter] = None
    last_prediction: Optional[Tuple[float, float]] = None
    
    # Track statistics
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    max_area: float = 0.0
    avg_area: float = 0.0
    
    def add_detection(self, detection: DefectDetection):
        """Add new detection to track and update statistics"""
        self.detections.append(detection)
        self.age = 0  # Reset age since we have new detection
        
        # Update statistics
        confidences = [d.confidence for d in self.detections]
        areas = [d.area for d in self.detections]
        
        self.max_confidence = max(confidences)
        self.avg_confidence = sum(confidences) / len(confidences)
        self.max_area = max(areas)
        self.avg_area = sum(areas) / len(areas)
    
    @property
    def duration_frames(self) -> int:
        """Number of frames this track has existed"""
        return len(self.detections)
    
    @property
    def first_frame(self) -> int:
        """Frame where track first appeared"""
        return self.detections[0].frame_id if self.detections else -1
    
    @property
    def last_frame(self) -> int:
        """Frame where track was last seen"""
        return self.detections[-1].frame_id if self.detections else -1
    
    @property
    def current_centroid(self) -> Optional[Tuple[float, float]]:
        """Current centroid position"""
        return self.detections[-1].centroid if self.detections else None


class FabricByteTracker:
    """ByteTrack adapted for fabric defect tracking"""
    
    def __init__(self, 
                 initial_fabric_speed: float = 5.0,  # Now just an arbitrary starting point
                 high_conf_threshold: float = 0.7,
                 low_conf_threshold: float = 0.3,
                 max_lost_frames: int = 10,
                 iou_threshold: float = 0.3,
                 motion_penalty_weight: float = 0.2):
        """
        Initialize fabric defect tracker with adaptive speed estimation
        
        Args:
            initial_fabric_speed: Arbitrary starting speed (auto-adapts to real speed)
            high_conf_threshold: Threshold for high confidence detections
            low_conf_threshold: Threshold for low confidence detections  
            max_lost_frames: Max frames to keep lost tracks
            iou_threshold: IoU threshold for matching
            motion_penalty_weight: Weight for motion consistency penalty
        """
        # Initialize adaptive speed tracker
        self.speed_tracker = AdaptiveFabricSpeedTracker(
            initial_speed=initial_fabric_speed,
            bootstrap_frames=15,
            adaptation_window_size=30
        )
        
        # Base tracking parameters (may be adjusted based on fabric state)
        self.base_high_conf_thresh = high_conf_threshold
        self.base_low_conf_thresh = low_conf_threshold
        self.base_max_lost_frames = max_lost_frames
        self.base_iou_threshold = iou_threshold
        self.motion_penalty_weight = motion_penalty_weight
        
        self.tracks: List[DefectTrack] = []
        self.next_track_id = 1
        self.frame_count = 0
        
        # Performance tracking
        self.total_detections = 0
        self.total_tracks_created = 0
        
        logger.info(f"FabricByteTracker initialized with adaptive speed tracking (initial={initial_fabric_speed})")
        
    def update(self, detections: List[DefectDetection], 
               fabric_motion: Tuple[float, float] = (0, 0)) -> List[DefectTrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of defect detections from current frame
            fabric_motion: (dx, dy) fabric movement since last frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        self.total_detections += len(detections)
        
        # Update fabric speed estimate using adaptive speed tracker
        if fabric_motion[0] != 0 or fabric_motion[1] != 0:
            motion_speed = np.linalg.norm(fabric_motion)
            motion_confidence = min(1.0, motion_speed / max(self.speed_tracker.get_current_speed(), 1.0))
            
            # Update adaptive speed tracker
            self.speed_tracker.update_speed(
                new_speed=motion_speed,
                confidence=motion_confidence,
                method_used="fabric_motion_estimation"
            )
        
        # Get current adaptive parameters based on fabric state
        tracking_params = self.speed_tracker.get_tracking_parameters()
        self.high_conf_thresh = tracking_params.get('high_conf_threshold', self.base_high_conf_thresh)
        self.low_conf_thresh = tracking_params.get('low_conf_threshold', self.base_low_conf_thresh)
        self.max_lost_frames = tracking_params.get('max_lost_frames', self.base_max_lost_frames)
        self.iou_threshold = tracking_params.get('iou_threshold', self.base_iou_threshold)
        
        logger.debug(f"Frame {self.frame_count}: Processing {len(detections)} detections, "
                    f"fabric_motion={fabric_motion}, speed={self.speed_tracker.get_current_speed():.2f}, "
                    f"state={self.speed_tracker.get_fabric_state().value}")
        
        # Predict track positions using fabric motion
        self._predict_tracks(fabric_motion)
        
        # Separate high and low confidence detections (ByteTrack strategy)
        high_conf_dets = [d for d in detections if d.confidence >= self.high_conf_thresh]
        low_conf_dets = [d for d in detections if self.low_conf_thresh <= d.confidence < self.high_conf_thresh]
        
        logger.debug(f"High conf: {len(high_conf_dets)}, Low conf: {len(low_conf_dets)}")
        
        # First association: high confidence detections with active tracks
        active_tracks = [t for t in self.tracks if t.state == 'active']
        matched_pairs, unmatched_tracks, unmatched_high_dets = self._associate(active_tracks, high_conf_dets)
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            self._update_track(active_tracks[track_idx], high_conf_dets[det_idx])
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            active_tracks[track_idx].state = 'lost'
            active_tracks[track_idx].age += 1
        
        # Second association: low confidence detections with lost tracks
        lost_tracks = [t for t in self.tracks if t.state == 'lost' and t.age <= self.max_lost_frames]
        matched_pairs2, unmatched_lost, unmatched_low_dets = self._associate(lost_tracks, low_conf_dets)
        
        # Recover lost tracks
        for track_idx, det_idx in matched_pairs2:
            self._update_track(lost_tracks[track_idx], low_conf_dets[det_idx])
            lost_tracks[track_idx].state = 'active'
        
        # Create new tracks from unmatched high confidence detections
        for det_idx in unmatched_high_dets:
            self._create_new_track(high_conf_dets[det_idx])
        
        # Remove old lost tracks
        old_track_count = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.age <= self.max_lost_frames or t.state != 'lost']
        removed_tracks = old_track_count - len(self.tracks)
        
        if removed_tracks > 0:
            logger.debug(f"Removed {removed_tracks} old tracks")
        
        active_tracks = [t for t in self.tracks if t.state == 'active']
        logger.debug(f"Active tracks: {len(active_tracks)}, Total tracks: {len(self.tracks)}")
        
        return active_tracks
    
    def _predict_tracks(self, fabric_motion: Tuple[float, float]):
        """Predict track positions using Kalman filter + fabric motion"""
        for track in self.tracks:
            if track.kalman_filter is not None:
                # Standard Kalman prediction
                prediction = track.kalman_filter.predict()
                
                # Add fabric motion bias - handle both tuple and array inputs
                if hasattr(fabric_motion, '__len__') and len(fabric_motion) >= 2:
                    dx, dy = fabric_motion[0], fabric_motion[1]
                else:
                    dx, dy = 0, 0  # Fallback if motion data is invalid
                
                # Handle prediction array indexing safely
                pred_x = float(prediction[0] if prediction.ndim == 1 else prediction[0, 0])
                pred_y = float(prediction[1] if prediction.ndim == 1 else prediction[1, 0])
                
                track.last_prediction = (pred_x + dx, pred_y + dy)
            elif track.current_centroid:
                # Fallback: simple motion prediction
                if hasattr(fabric_motion, '__len__') and len(fabric_motion) >= 2:
                    dx, dy = fabric_motion[0], fabric_motion[1]
                else:
                    dx, dy = 0, 0
                    
                track.last_prediction = (
                    track.current_centroid[0] + dx,
                    track.current_centroid[1] + dy
                )
            
            track.age += 1
    
    def _associate(self, tracks: List[DefectTrack], 
                  detections: List[DefectDetection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU + fabric motion"""
        
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix (lower cost = better match)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost = self._calculate_association_cost(track, det)
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm for optimal assignment
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        
        # Filter matches by cost threshold
        max_cost = 1.0 - self.iou_threshold + self.motion_penalty_weight
        for i, j in zip(matched_indices[0], matched_indices[1]):
            if cost_matrix[i, j] < max_cost:
                matched_pairs.append((i, j))
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in [pair[0] for pair in matched_pairs]]
        unmatched_dets = [j for j in range(len(detections)) if j not in [pair[1] for pair in matched_pairs]]
        
        return matched_pairs, unmatched_tracks, unmatched_dets
    
    def _calculate_association_cost(self, track: DefectTrack, detection: DefectDetection) -> float:
        """Calculate cost for associating track with detection"""
        # Get predicted position
        pred_pos = track.last_prediction or track.current_centroid
        if pred_pos is None:
            return 1.0  # Maximum cost if no position available
        
        # Calculate IoU with predicted position
        if track.detections:
            # Use last detection bbox size for prediction
            last_bbox = track.detections[-1].bbox
            pred_bbox = self._centroid_to_bbox(pred_pos, (last_bbox[2], last_bbox[3]))
        else:
            # Fallback to detection bbox
            pred_bbox = detection.bbox
        
        iou = self._calculate_iou(pred_bbox, detection.bbox)
        
        # Calculate motion consistency penalty
        motion_penalty = self._calculate_motion_penalty(track, detection)
        
        # Combined cost (lower is better)
        cost = (1 - iou) + self.motion_penalty_weight * motion_penalty
        
        return cost
    
    def _calculate_motion_penalty(self, track: DefectTrack, detection: DefectDetection) -> float:
        """Calculate penalty based on motion inconsistency with fabric flow"""
        if len(track.detections) < 2:
            return 0.0
        
        # Expected position based on fabric motion
        last_pos = track.current_centroid
        if last_pos is None:
            return 0.0
        
        current_fabric_speed = self.speed_tracker.get_current_speed()
        expected_pos = (
            last_pos[0] + current_fabric_speed,  # Assuming horizontal fabric movement
            last_pos[1]  # Vertical position should remain similar
        )
        
        # Actual detection position
        actual_pos = detection.centroid
        
        # Distance penalty (normalized)
        distance = np.linalg.norm(np.array(expected_pos) - np.array(actual_pos))
        
        # Normalize by expected movement (fabric speed)
        normalized_penalty = min(distance / max(current_fabric_speed, 10.0), 1.0)
        
        return normalized_penalty
    
    def _create_new_track(self, detection: DefectDetection):
        """Create new track from detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        self.total_tracks_created += 1
        
        # Initialize Kalman filter for fabric motion
        kalman = cv2.KalmanFilter(4, 2)  # 4 states (x,y,vx,vy), 2 measurements (x,y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        
        # Initialize state with current fabric speed assumption
        current_fabric_speed = self.speed_tracker.get_current_speed()
        initial_state = np.array([
            detection.centroid[0], 
            detection.centroid[1], 
            current_fabric_speed, 
            0
        ], dtype=np.float32)
        
        kalman.statePre = initial_state
        kalman.statePost = initial_state.copy()
        
        track = DefectTrack(
            track_id=track_id,
            state='active',
            age=0,
            fabric_start_position=detection.centroid[0],  # Assuming horizontal fabric movement
            kalman_filter=kalman,
            last_prediction=None
        )
        
        track.add_detection(detection)
        self.tracks.append(track)
        
        logger.debug(f"Created new track {track_id} at position {detection.centroid}")
    
    def _update_track(self, track: DefectTrack, detection: DefectDetection):
        """Update existing track with new detection"""
        track.add_detection(detection)
        
        # Update Kalman filter
        if track.kalman_filter is not None:
            measurement = np.array([[detection.centroid[0]], [detection.centroid[1]]], dtype=np.float32)
            track.kalman_filter.correct(measurement)
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        intersection = wi * hi
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _centroid_to_bbox(self, centroid: Tuple[float, float], 
                         size: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Convert centroid to bounding box"""
        w, h = size
        x = centroid[0] - w/2
        y = centroid[1] - h/2
        return (x, y, w, h)
    
    def get_completed_tracks(self) -> List[DefectTrack]:
        """Get tracks that are no longer active (completed lifecycle)"""
        return [t for t in self.tracks if t.state == 'completed']
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = [t for t in self.tracks if t.state == 'active']
        speed_stats = self.speed_tracker.get_statistics()
        return {
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': len(active_tracks),
            'total_tracks': len(self.tracks),
            'avg_detections_per_frame': self.total_detections / max(self.frame_count, 1),
            'current_fabric_speed': speed_stats['current_speed'],
            'fabric_state': speed_stats['fabric_state'],
            'speed_confidence': speed_stats['speed_confidence'],
            'is_speed_bootstrapped': speed_stats['is_bootstrapped']
        }