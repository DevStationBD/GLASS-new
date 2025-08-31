#!/usr/bin/env python3
"""
Fabric Motion Estimation for Defect Tracking

This module provides various methods to estimate fabric movement between consecutive frames,
which is crucial for accurate defect tracking in continuous fabric inspection.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MotionEstimationMethod(Enum):
    """Available motion estimation methods"""
    OPTICAL_FLOW_DENSE = "optical_flow_dense"
    OPTICAL_FLOW_SPARSE = "optical_flow_sparse" 
    TEMPLATE_MATCHING = "template_matching"
    PHASE_CORRELATION = "phase_correlation"
    FEATURE_MATCHING = "feature_matching"

@dataclass
class MotionEstimate:
    """Motion estimation result"""
    displacement: Tuple[float, float]  # (dx, dy) in pixels
    confidence: float  # Confidence score 0-1
    speed_pixels_per_frame: float  # Movement magnitude
    angle_degrees: float  # Movement direction
    method_used: MotionEstimationMethod
    additional_info: Dict  # Method-specific additional information

class FabricMotionEstimator:
    """Estimates fabric movement between consecutive frames"""
    
    def __init__(self, 
                 method: MotionEstimationMethod = MotionEstimationMethod.OPTICAL_FLOW_SPARSE,
                 fabric_roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize motion estimator
        
        Args:
            method: Motion estimation method to use
            fabric_roi: Region of interest (x, y, w, h) for fabric area, None for full frame
        """
        self.method = method
        self.fabric_roi = fabric_roi
        self.prev_frame = None
        self.prev_features = None
        
        # Method-specific parameters
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Motion history for smoothing
        self.motion_history: List[Tuple[float, float]] = []
        self.history_size = 5
        
        logger.info(f"FabricMotionEstimator initialized with method: {method.value}")
    
    def estimate_motion(self, current_frame: np.ndarray) -> MotionEstimate:
        """
        Estimate fabric motion from previous to current frame
        
        Args:
            current_frame: Current frame (H, W, 3) or (H, W)
            
        Returns:
            MotionEstimate object with movement information
        """
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_current = current_frame.copy()
        
        # Extract ROI if specified
        if self.fabric_roi:
            x, y, w, h = self.fabric_roi
            gray_current = gray_current[y:y+h, x:x+w]
        
        # If no previous frame, return zero motion
        if self.prev_frame is None:
            self.prev_frame = gray_current.copy()
            return MotionEstimate(
                displacement=(0.0, 0.0),
                confidence=0.0,
                speed_pixels_per_frame=0.0,
                angle_degrees=0.0,
                method_used=self.method,
                additional_info={}
            )
        
        # Estimate motion based on selected method
        if self.method == MotionEstimationMethod.OPTICAL_FLOW_SPARSE:
            result = self._estimate_optical_flow_sparse(gray_current)
        elif self.method == MotionEstimationMethod.OPTICAL_FLOW_DENSE:
            result = self._estimate_optical_flow_dense(gray_current)
        elif self.method == MotionEstimationMethod.TEMPLATE_MATCHING:
            result = self._estimate_template_matching(gray_current)
        elif self.method == MotionEstimationMethod.PHASE_CORRELATION:
            result = self._estimate_phase_correlation(gray_current)
        elif self.method == MotionEstimationMethod.FEATURE_MATCHING:
            result = self._estimate_feature_matching(gray_current)
        else:
            logger.warning(f"Unknown method {self.method}, using sparse optical flow")
            result = self._estimate_optical_flow_sparse(gray_current)
        
        # Apply temporal smoothing
        smoothed_result = self._apply_temporal_smoothing(result)
        
        # Update state
        self.prev_frame = gray_current.copy()
        
        return smoothed_result
    
    def _estimate_optical_flow_sparse(self, current_frame: np.ndarray) -> MotionEstimate:
        """Estimate motion using sparse optical flow (Lucas-Kanade)"""
        try:
            # Detect features in previous frame if not available
            if self.prev_features is None:
                self.prev_features = cv2.goodFeaturesToTrack(self.prev_frame, **self.feature_params)
            
            if self.prev_features is None or len(self.prev_features) == 0:
                # No features found, try to detect in current frame
                features = cv2.goodFeaturesToTrack(current_frame, **self.feature_params)
                self.prev_features = features
                return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"features_found": 0})
            
            # Calculate optical flow
            next_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_frame, self.prev_features, None, **self.optical_flow_params
            )
            
            # Select good features
            good_new = next_features[status == 1]
            good_old = self.prev_features[status == 1]
            
            if len(good_new) < 5:  # Not enough good features
                self.prev_features = cv2.goodFeaturesToTrack(current_frame, **self.feature_params)
                return MotionEstimate((0.0, 0.0), 0.1, 0.0, 0.0, self.method, {"features_tracked": len(good_new)})
            
            # Calculate displacement vectors
            displacements = good_new - good_old
            
            # Calculate median displacement (robust to outliers)
            median_dx = float(np.median(displacements[:, 0]))
            median_dy = float(np.median(displacements[:, 1]))
            
            # Calculate confidence based on consistency of motion vectors
            displacement_std = np.std(displacements, axis=0)
            consistency = 1.0 / (1.0 + np.linalg.norm(displacement_std))
            confidence = min(consistency * len(good_new) / 50.0, 1.0)
            
            # Calculate speed and angle
            speed = np.linalg.norm([median_dx, median_dy])
            angle = np.degrees(np.arctan2(median_dy, median_dx))
            
            # Update features for next frame
            self.prev_features = good_new.reshape(-1, 1, 2)
            
            return MotionEstimate(
                displacement=(median_dx, median_dy),
                confidence=confidence,
                speed_pixels_per_frame=speed,
                angle_degrees=angle,
                method_used=self.method,
                additional_info={
                    "features_tracked": len(good_new),
                    "features_total": len(self.prev_features),
                    "displacement_std": displacement_std.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in sparse optical flow: {e}")
            return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": str(e)})
    
    def _estimate_optical_flow_dense(self, current_frame: np.ndarray) -> MotionEstimate:
        """Estimate motion using dense optical flow (Farneback)"""
        try:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_frame, None, None)
            
            # Alternative: use Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_frame, None, 
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Filter out low magnitude flows (noise)
            valid_mask = magnitude > 1.0
            
            if np.sum(valid_mask) < 100:  # Not enough valid flow
                return MotionEstimate((0.0, 0.0), 0.1, 0.0, 0.0, self.method, {"valid_pixels": int(np.sum(valid_mask))})
            
            # Calculate median flow
            valid_flow = flow[valid_mask]
            median_dx = float(np.median(valid_flow[:, 0]))
            median_dy = float(np.median(valid_flow[:, 1]))
            
            # Calculate confidence based on flow consistency
            flow_std = np.std(valid_flow, axis=0)
            consistency = 1.0 / (1.0 + np.linalg.norm(flow_std))
            confidence = min(consistency, 1.0)
            
            # Calculate speed and angle
            speed = np.linalg.norm([median_dx, median_dy])
            motion_angle = np.degrees(np.arctan2(median_dy, median_dx))
            
            return MotionEstimate(
                displacement=(median_dx, median_dy),
                confidence=confidence,
                speed_pixels_per_frame=speed,
                angle_degrees=motion_angle,
                method_used=self.method,
                additional_info={
                    "valid_pixels": int(np.sum(valid_mask)),
                    "total_pixels": flow.shape[0] * flow.shape[1],
                    "flow_std": flow_std.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in dense optical flow: {e}")
            return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": str(e)})
    
    def _estimate_template_matching(self, current_frame: np.ndarray) -> MotionEstimate:
        """Estimate motion using template matching"""
        try:
            # Use center region of previous frame as template
            h, w = self.prev_frame.shape
            template_size = min(h // 4, w // 4, 100)
            center_x, center_y = w // 2, h // 2
            
            template = self.prev_frame[
                center_y - template_size//2:center_y + template_size//2,
                center_x - template_size//2:center_x + template_size//2
            ]
            
            if template.size == 0:
                return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": "empty_template"})
            
            # Template matching
            result = cv2.matchTemplate(current_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate displacement
            dx = max_loc[0] - (center_x - template_size//2)
            dy = max_loc[1] - (center_y - template_size//2)
            
            # Confidence based on match quality
            confidence = float(max_val) if max_val > 0 else 0.0
            
            # Calculate speed and angle
            speed = np.linalg.norm([dx, dy])
            angle = np.degrees(np.arctan2(dy, dx))
            
            return MotionEstimate(
                displacement=(float(dx), float(dy)),
                confidence=confidence,
                speed_pixels_per_frame=speed,
                angle_degrees=angle,
                method_used=self.method,
                additional_info={
                    "match_score": float(max_val),
                    "template_size": template_size
                }
            )
            
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": str(e)})
    
    def _estimate_phase_correlation(self, current_frame: np.ndarray) -> MotionEstimate:
        """Estimate motion using phase correlation"""
        try:
            # Ensure frames are the same size
            if self.prev_frame.shape != current_frame.shape:
                current_frame = cv2.resize(current_frame, (self.prev_frame.shape[1], self.prev_frame.shape[0]))
            
            # Convert to float32
            prev_float = self.prev_frame.astype(np.float32)
            curr_float = current_frame.astype(np.float32)
            
            # Calculate phase correlation
            (dx, dy), response = cv2.phaseCorrelate(prev_float, curr_float)
            
            # Calculate speed and angle
            speed = np.linalg.norm([dx, dy])
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Confidence based on response value
            confidence = min(float(response), 1.0) if response > 0 else 0.0
            
            return MotionEstimate(
                displacement=(float(dx), float(dy)),
                confidence=confidence,
                speed_pixels_per_frame=speed,
                angle_degrees=angle,
                method_used=self.method,
                additional_info={
                    "phase_correlation_response": float(response)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in phase correlation: {e}")
            return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": str(e)})
    
    def _estimate_feature_matching(self, current_frame: np.ndarray) -> MotionEstimate:
        """Estimate motion using feature matching (SIFT/ORB)"""
        try:
            # Initialize ORB detector (faster than SIFT)
            orb = cv2.ORB_create(nfeatures=500)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(self.prev_frame, None)
            kp2, des2 = orb.detectAndCompute(current_frame, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return MotionEstimate((0.0, 0.0), 0.1, 0.0, 0.0, self.method, {"keypoints": (len(kp1), len(kp2))})
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use top matches
            good_matches = matches[:min(50, len(matches))]
            
            if len(good_matches) < 10:
                return MotionEstimate((0.0, 0.0), 0.1, 0.0, 0.0, self.method, {"good_matches": len(good_matches)})
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            # Calculate displacements
            displacements = dst_pts - src_pts
            
            # Calculate median displacement
            median_dx = float(np.median(displacements[:, 0]))
            median_dy = float(np.median(displacements[:, 1]))
            
            # Calculate confidence based on displacement consistency
            displacement_std = np.std(displacements, axis=0)
            consistency = 1.0 / (1.0 + np.linalg.norm(displacement_std))
            confidence = min(consistency, 1.0)
            
            # Calculate speed and angle
            speed = np.linalg.norm([median_dx, median_dy])
            angle = np.degrees(np.arctan2(median_dy, median_dx))
            
            return MotionEstimate(
                displacement=(median_dx, median_dy),
                confidence=confidence,
                speed_pixels_per_frame=speed,
                angle_degrees=angle,
                method_used=self.method,
                additional_info={
                    "matches_total": len(matches),
                    "matches_good": len(good_matches),
                    "keypoints": (len(kp1), len(kp2)),
                    "displacement_std": displacement_std.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in feature matching: {e}")
            return MotionEstimate((0.0, 0.0), 0.0, 0.0, 0.0, self.method, {"error": str(e)})
    
    def _apply_temporal_smoothing(self, current_estimate: MotionEstimate) -> MotionEstimate:
        """Apply temporal smoothing to motion estimates"""
        # Add current displacement to history
        self.motion_history.append(current_estimate.displacement)
        
        # Keep only recent history
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)
        
        # Calculate smoothed displacement
        if len(self.motion_history) > 1:
            history_array = np.array(self.motion_history)
            smoothed_dx = float(np.median(history_array[:, 0]))
            smoothed_dy = float(np.median(history_array[:, 1]))
            
            # Blend current estimate with history
            alpha = 0.7  # Weight for current estimate
            final_dx = alpha * current_estimate.displacement[0] + (1 - alpha) * smoothed_dx
            final_dy = alpha * current_estimate.displacement[1] + (1 - alpha) * smoothed_dy
        else:
            final_dx, final_dy = current_estimate.displacement
        
        # Create smoothed result
        speed = np.linalg.norm([final_dx, final_dy])
        angle = np.degrees(np.arctan2(final_dy, final_dx))
        
        smoothed_info = current_estimate.additional_info.copy()
        smoothed_info.update({
            "original_displacement": current_estimate.displacement,
            "smoothing_history_size": len(self.motion_history)
        })
        
        return MotionEstimate(
            displacement=(final_dx, final_dy),
            confidence=current_estimate.confidence,
            speed_pixels_per_frame=speed,
            angle_degrees=angle,
            method_used=current_estimate.method_used,
            additional_info=smoothed_info
        )
    
    def reset(self):
        """Reset estimator state"""
        self.prev_frame = None
        self.prev_features = None
        self.motion_history = []
        logger.info("FabricMotionEstimator reset")
    
    def set_fabric_roi(self, roi: Tuple[int, int, int, int]):
        """Set fabric region of interest"""
        self.fabric_roi = roi
        logger.info(f"Fabric ROI set to: {roi}")
    
    def get_average_motion(self) -> Tuple[float, float]:
        """Get average motion from recent history"""
        if not self.motion_history:
            return (0.0, 0.0)
        
        history_array = np.array(self.motion_history)
        avg_dx = float(np.mean(history_array[:, 0]))
        avg_dy = float(np.mean(history_array[:, 1]))
        
        return (avg_dx, avg_dy)