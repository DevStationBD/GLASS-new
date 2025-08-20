"""
Size Analyzer Module for GLASS Framework

This module provides defect size analysis capabilities including:
- Physical size measurements 
- Defect area calculations
- Statistical analysis of defects
- Multi-unit support (mm, cm, inch)
"""

from .defect_size_analyzer import DefectSizeAnalyzer, DefectMetrics

__all__ = ['DefectSizeAnalyzer', 'DefectMetrics']