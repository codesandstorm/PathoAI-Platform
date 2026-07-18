"""
PathoAI-Platform
================
Research-grade computational pathology platform for automated
stromal TIL (sTIL) scoring from Whole Slide Images (WSIs).

Implements the sTIL scoring methodology from Salgado et al. (2015)
using the TIGER public dataset for training and validation.

Authors: PathoAI Research Team
Created: 2026-07-18
Milestone: 1 — Infrastructure
"""

__version__ = "0.1.0"
__author__ = "PathoAI Research Team"
__email__ = "research@pathoai.dev"
__license__ = "MIT"
__status__ = "Research"

# Minimum Python version enforcement
import sys
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"PathoAI-Platform requires Python >= 3.10. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )
