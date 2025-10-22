"""
Defect Tracking Module for GLASS Fabric Inspection

This module provides temporal defect tracking capabilities for continuous fabric inspection,
enabling tracking of defects across multiple frames as fabric moves through the system.
"""

from .fabric_bytetrack import FabricByteTracker, DefectDetection, DefectTrack
from .glass_integration import GLASSDefectTracker, TrackedDefectSummary
from .motion_estimation import FabricMotionEstimator, MotionEstimationMethod, MotionEstimate

__all__ = [
    'FabricByteTracker',
    'DefectDetection', 
    'DefectTrack',
    'GLASSDefectTracker',
    'TrackedDefectSummary',
    'FabricMotionEstimator',
    'MotionEstimationMethod',
    'MotionEstimate'
]

__version__ = "1.0.0"