from .config import LLMConfig, Problem, LLM_PROVIDERS
from .models import (
    RolePreference, Solution, PeerReview, Error,
    RefinedSolution, Judgment, Assessment, Severity, ErrorType
)
from .main import DebateSystem

__version__ = "1.0.0"
__all__ = [
    "DebateSystem",
    "LLMConfig",
    "Problem",
    "RolePreference",
    "Solution",
    "PeerReview",
    "RefinedSolution",
    "Judgment",
    "Assessment",
    "Severity",
    "ErrorType",
    "Error"
]