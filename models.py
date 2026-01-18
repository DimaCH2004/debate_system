from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class ErrorType(Enum):
    LOGICAL_ERROR = "logical_error"
    CALCULATION_ERROR = "calculation_error"
    ASSUMPTION_ERROR = "assumption_error"
    MISSING_CASE = "missing_case"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Assessment(Enum):
    CORRECT = "correct"
    PROMISING_BUT_FLAWED = "promising_but_flawed"
    FUNDAMENTALLY_FLAWED = "fundamentally_flawed"
    INCOMPLETE = "incomplete"


@dataclass
class Error:
    location: str
    error_type: ErrorType
    description: str
    severity: Severity
    suggested_fix: str = ""

    def to_dict(self):
        return {
            "location": self.location,
            "error_type": self.error_type.value,
            "description": self.description,
            "severity": self.severity.value,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class RolePreference:
    preferred_roles: List[str]
    confidence_by_role: Dict[str, float]
    reasoning: str
    self_assessment: str = ""

    def to_dict(self):
        return {
            "preferred_roles": self.preferred_roles,
            "confidence_by_role": self.confidence_by_role,
            "reasoning": self.reasoning,
            "self_assessment": self.self_assessment
        }


@dataclass
class Solution:
    solver_id: str
    solution_text: str
    final_answer: str
    confidence: float
    reasoning_steps: List[str]
    assumptions: List[str]

    def to_dict(self):
        return {
            "solver_id": self.solver_id,
            "solution_text": self.solution_text,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "assumptions": self.assumptions
        }


@dataclass
class PeerReview:
    reviewer_id: str
    solution_id: str
    strengths: List[str]
    weaknesses: List[str]
    errors: List[Error]
    suggested_changes: List[str]
    overall_assessment: Assessment
    confidence: float

    def to_dict(self):
        return {
            "reviewer_id": self.reviewer_id,
            "solution_id": self.solution_id,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "errors": [e.to_dict() for e in self.errors],
            "suggested_changes": self.suggested_changes,
            "overall_assessment": self.overall_assessment.value,
            "confidence": self.confidence
        }


@dataclass
class CritiqueResponse:
    critique_id: str
    accepted: bool
    response: str
    changes_made: str

    def to_dict(self):
        return {
            "critique_id": self.critique_id,
            "accepted": self.accepted,
            "response": self.response,
            "changes_made": self.changes_made
        }


@dataclass
class RefinedSolution:
    original_solution_id: str
    changes_made: List[CritiqueResponse]
    refined_solution: str
    refined_answer: str
    confidence: float

    def to_dict(self):
        return {
            "original_solution_id": self.original_solution_id,
            "changes_made": [c.to_dict() for c in self.changes_made],
            "refined_solution": self.refined_solution,
            "refined_answer": self.refined_answer,
            "confidence": self.confidence
        }