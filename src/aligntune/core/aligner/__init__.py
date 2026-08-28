"""
AlignTune Aligner: Interactive Training Inspector

Provides live, interactive training session inspection with:
- AlignerSession: Python API for trainer control
- AlignerCallback: Trainer integration hook
- AlignerDashboard: Terminal UI with rich
"""

from .session import AlignerSession, AlignerState
from .callback import AlignerCallback
from .dashboard import AlignerDashboard, create_dashboard

__all__ = [
    "AlignerSession",
    "AlignerState",
    "AlignerCallback",
    "AlignerDashboard",
    "create_dashboard",
]
