#!/usr/bin/env python3
"""
Adaptive Fabric Speed Estimation for Dynamic Tracking

This module provides robust, automatic fabric speed estimation without requiring
manual configuration. It adapts to variable speeds and handles industrial scenarios
like startup/shutdown, speed changes, and brief stops.
"""

import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FabricState(Enum):
    """Current fabric movement state"""
    INITIALIZING = "initializing"
    STEADY_MOVING = "steady_moving"
    VARIABLE_SPEED = "variable_speed"
    STOPPED = "stopped"
    STARTING_UP = "starting_up"

@dataclass
class SpeedEstimate:
    """Single speed estimation result"""
    speed_pixels_per_frame: float
    confidence: float
    method_used: str
    timestamp: int  # Frame number

class AdaptiveFabricSpeedTracker:
    """
    Automatic fabric speed tracking with no manual configuration required.
    
    Starts with an arbitrary initial value and adapts to real fabric motion
    through robust estimation and outlier rejection.
    """
    
    def __init__(self, 
                 initial_speed: float = 5.0,  # Arbitrary starting point
                 bootstrap_frames: int = 15,
                 adaptation_window_size: int = 30):
        """
        Initialize adaptive speed tracker
        
        Args:
            initial_speed: Arbitrary starting speed (will be overridden quickly)
            bootstrap_frames: Number of frames for initial speed estimation
            adaptation_window_size: Sliding window size for speed adaptation
        """
        # Starting values - will adapt automatically
        self.current_speed = initial_speed
        self.initial_speed = initial_speed
        
        # Speed estimation history
        self.speed_history = deque(maxlen=adaptation_window_size)
        self.confidence_history = deque(maxlen=adaptation_window_size)
        
        # Adaptation parameters
        self.bootstrap_frames = bootstrap_frames
        self.frame_count = 0
        self.is_bootstrapped = False
        
        # State tracking
        self.fabric_state = FabricState.INITIALIZING
        self.speed_variance = 0.0
        self.adaptation_rate = 0.2
        
        # Thresholds
        self.min_confidence = 0.4
        self.outlier_threshold = 3.0  # Modified z-score threshold
        self.steady_state_variance_threshold = 0.5
        self.stopped_speed_threshold = 1.0
        
        logger.info(f"AdaptiveFabricSpeedTracker initialized with arbitrary initial_speed={initial_speed}")
    
    def update_speed(self, 
                     new_speed: float, 
                     confidence: float, 
                     method_used: str = "motion_estimation") -> bool:
        """
        Update fabric speed with new estimate
        
        Args:
            new_speed: New speed estimate in pixels per frame
            confidence: Confidence score (0-1)
            method_used: Method that generated this estimate
            
        Returns:
            bool: True if estimate was accepted, False if rejected as outlier
        """
        self.frame_count += 1
        
        # Skip low confidence estimates
        if confidence < self.min_confidence:
            logger.debug(f"Frame {self.frame_count}: Rejecting low confidence speed estimate: "
                        f"speed={new_speed:.2f}, confidence={confidence:.2f}")
            return False
        
        # Bootstrap phase - be more accepting initially
        if not self.is_bootstrapped:
            return self._bootstrap_update(new_speed, confidence, method_used)
        
        # Steady state - apply outlier detection
        return self._steady_state_update(new_speed, confidence, method_used)
    
    def _bootstrap_update(self, new_speed: float, confidence: float, method_used: str) -> bool:
        """Handle speed updates during bootstrap phase"""
        
        self.speed_history.append(new_speed)
        self.confidence_history.append(confidence)
        
        # Use weighted running average during bootstrap
        speeds = np.array(list(self.speed_history))
        confidences = np.array(list(self.confidence_history))
        
        if len(speeds) >= 3:
            # Weighted average by confidence
            self.current_speed = np.average(speeds, weights=confidences)
        else:
            # Simple average for first few frames
            self.current_speed = np.mean(speeds)
        
        # Check if we have enough data to finish bootstrap
        if len(self.speed_history) >= self.bootstrap_frames:
            self.is_bootstrapped = True
            self.speed_variance = np.var(speeds)
            self._update_fabric_state()
            
            logger.info(f"Bootstrap complete after {self.frame_count} frames: "
                       f"speed={self.current_speed:.2f}, variance={self.speed_variance:.2f}")
        
        logger.debug(f"Bootstrap frame {self.frame_count}: speed={new_speed:.2f} -> "
                    f"current={self.current_speed:.2f}")
        
        return True
    
    def _steady_state_update(self, new_speed: float, confidence: float, method_used: str) -> bool:
        """Handle speed updates during steady state with outlier detection"""
        
        # Calculate current statistics
        speeds = np.array(list(self.speed_history))
        median_speed = np.median(speeds)
        mad = np.median(np.abs(speeds - median_speed))  # Median Absolute Deviation
        
        # Modified z-score for outlier detection (more robust than standard deviation)
        if mad > 0:
            modified_z_score = 0.6745 * (new_speed - median_speed) / mad
        else:
            modified_z_score = 0
        
        # Check if this is an outlier
        is_outlier = abs(modified_z_score) > self.outlier_threshold
        
        if is_outlier:
            logger.debug(f"Frame {self.frame_count}: Rejecting speed outlier: "
                        f"speed={new_speed:.2f}, median={median_speed:.2f}, "
                        f"z_score={modified_z_score:.2f}")
            return False
        
        # Accept the estimate
        self.speed_history.append(new_speed)
        self.confidence_history.append(confidence)
        
        # Adaptive speed update with confidence weighting
        alpha = min(0.4, confidence * self.adaptation_rate)  # Cap adaptation rate
        self.current_speed = alpha * new_speed + (1 - alpha) * self.current_speed
        
        # Update statistics and state
        self.speed_variance = np.var(speeds)
        self._update_fabric_state()
        
        logger.debug(f"Frame {self.frame_count}: Accepted speed: {new_speed:.2f} -> "
                    f"current={self.current_speed:.2f}, state={self.fabric_state.value}")
        
        return True
    
    def _update_fabric_state(self):
        """Update fabric state based on current speed and variance"""
        
        if not self.is_bootstrapped:
            self.fabric_state = FabricState.INITIALIZING
            self.adaptation_rate = 0.3  # Fast adaptation during initialization
            return
        
        # Determine state based on speed and variance
        if self.current_speed < self.stopped_speed_threshold:
            self.fabric_state = FabricState.STOPPED
            self.adaptation_rate = 0.4  # Fast adaptation when starting up again
            
        elif self.speed_variance > self.current_speed * self.steady_state_variance_threshold:
            if self.current_speed > self.initial_speed * 0.5:
                self.fabric_state = FabricState.VARIABLE_SPEED
                self.adaptation_rate = 0.3  # Moderate adaptation for variable conditions
            else:
                self.fabric_state = FabricState.STARTING_UP  
                self.adaptation_rate = 0.4  # Fast adaptation during startup
                
        else:
            self.fabric_state = FabricState.STEADY_MOVING
            self.adaptation_rate = 0.1  # Slow adaptation for steady state
    
    def get_current_speed(self) -> float:
        """Get current fabric speed estimate"""
        return self.current_speed
    
    def get_fabric_state(self) -> FabricState:
        """Get current fabric state"""
        return self.fabric_state
    
    def get_speed_confidence(self) -> float:
        """Get confidence in current speed estimate"""
        if not self.is_bootstrapped or len(self.confidence_history) == 0:
            return 0.5  # Medium confidence during bootstrap
        
        # Confidence based on recent estimate quality and stability
        recent_confidence = np.mean(list(self.confidence_history)[-5:])  # Last 5 frames
        stability_factor = 1.0 / (1.0 + self.speed_variance)  # Higher variance = lower confidence
        
        return min(1.0, recent_confidence * stability_factor)
    
    def get_tracking_parameters(self) -> dict:
        """Get recommended tracking parameters based on current fabric state"""
        
        base_params = {
            'fabric_speed': self.current_speed,
            'high_conf_threshold': 0.7,
            'low_conf_threshold': 0.3,
            'max_lost_frames': 10,
            'iou_threshold': 0.5
        }
        
        # Adjust parameters based on fabric state
        if self.fabric_state == FabricState.STOPPED:
            base_params.update({
                'high_conf_threshold': 0.5,  # Lower threshold when stopped
                'low_conf_threshold': 0.2,
                'max_lost_frames': 20  # Keep tracks longer when stopped
            })
        elif self.fabric_state == FabricState.VARIABLE_SPEED:
            base_params.update({
                'high_conf_threshold': 0.8,  # Higher threshold for variable conditions  
                'low_conf_threshold': 0.4,
                'iou_threshold': 0.3  # More lenient IoU for variable motion
            })
        elif self.fabric_state == FabricState.STARTING_UP:
            base_params.update({
                'high_conf_threshold': 0.6,  # Medium threshold during startup
                'low_conf_threshold': 0.3,
                'max_lost_frames': 15
            })
        
        return base_params
    
    def get_statistics(self) -> dict:
        """Get current tracking statistics"""
        return {
            'current_speed': self.current_speed,
            'initial_speed': self.initial_speed,
            'speed_variance': self.speed_variance,
            'fabric_state': self.fabric_state.value,
            'is_bootstrapped': self.is_bootstrapped,
            'frame_count': self.frame_count,
            'adaptation_rate': self.adaptation_rate,
            'speed_confidence': self.get_speed_confidence(),
            'estimates_count': len(self.speed_history)
        }
    
    def reset(self):
        """Reset tracker to initial state"""
        self.current_speed = self.initial_speed
        self.speed_history.clear()
        self.confidence_history.clear()
        self.frame_count = 0
        self.is_bootstrapped = False
        self.fabric_state = FabricState.INITIALIZING
        self.speed_variance = 0.0
        self.adaptation_rate = 0.2
        
        logger.info("AdaptiveFabricSpeedTracker reset to initial state")


class FabricMotionAnalyzer:
    """
    Analyzes fabric motion patterns to provide robust speed estimates
    """
    
    def __init__(self):
        self.prev_frame = None
        self.motion_buffer = deque(maxlen=10)  # Buffer recent motion estimates
    
    def analyze_frame_motion(self, current_frame: np.ndarray, 
                           motion_estimators: List) -> Optional[SpeedEstimate]:
        """
        Analyze motion using multiple estimators and return robust estimate
        
        Args:
            current_frame: Current video frame
            motion_estimators: List of motion estimation objects
            
        Returns:
            SpeedEstimate with robust motion data or None if insufficient data
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return None
        
        estimates = []
        
        # Get estimates from all available methods
        for estimator in motion_estimators:
            try:
                motion = estimator.estimate_motion(current_frame)
                if motion.confidence > 0.3:  # Basic confidence threshold
                    estimates.append(SpeedEstimate(
                        speed_pixels_per_frame=motion.speed_pixels_per_frame,
                        confidence=motion.confidence,
                        method_used=motion.method_used.value,
                        timestamp=len(self.motion_buffer)
                    ))
            except Exception as e:
                logger.warning(f"Motion estimation failed for {estimator}: {e}")
        
        if not estimates:
            self.prev_frame = current_frame.copy()
            return None
        
        # Calculate robust estimate from multiple methods
        robust_estimate = self._calculate_robust_estimate(estimates)
        
        self.motion_buffer.append(robust_estimate)
        self.prev_frame = current_frame.copy()
        
        return robust_estimate
    
    def _calculate_robust_estimate(self, estimates: List[SpeedEstimate]) -> SpeedEstimate:
        """Calculate robust speed estimate from multiple methods"""
        
        if len(estimates) == 1:
            return estimates[0]
        
        # Weighted average by confidence
        speeds = np.array([est.speed_pixels_per_frame for est in estimates])
        confidences = np.array([est.confidence for est in estimates])
        
        # Use median if estimates vary significantly (outlier protection)
        if np.std(speeds) > np.mean(speeds) * 0.5:
            robust_speed = np.median(speeds)
            robust_confidence = np.mean(confidences)
            logger.debug("Using median speed estimate due to high variance between methods")
        else:
            # Weighted average for consistent estimates
            robust_speed = np.average(speeds, weights=confidences)
            robust_confidence = np.mean(confidences)
        
        # Combine method names
        methods = [est.method_used for est in estimates]
        combined_method = f"robust_({'+'.join(methods)})"
        
        return SpeedEstimate(
            speed_pixels_per_frame=robust_speed,
            confidence=robust_confidence,
            method_used=combined_method,
            timestamp=estimates[0].timestamp
        )