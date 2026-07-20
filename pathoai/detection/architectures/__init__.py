"""
pathoai/detection/architectures/__init__.py
============================================
Package trigger for registering object detector architectures.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.2
"""

from pathoai.detection.architectures.yolo import YOLODetector

__all__ = ["YOLODetector"]
