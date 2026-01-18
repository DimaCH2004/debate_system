from typing import Dict, List, Any
import json
import re
from models import PeerReview, Error, Severity, ErrorType, Assessment
from llm_client import LLMClient


class PeerReviewSystem:
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.llm_client = LLMClient()
        import logging
        self.logger = logging.getLogger(__name__)

    def generate_review(self, reviewer_id: str, solution: Dict, problem: str) -> PeerReview:
        prompt = f"""
        Problem: {problem}

        Solution to review:
        {json.dumps(solution, indent=2)}

        Provide structured critique:

        Strengths: [list 2-3 key strengths]
        Weaknesses: [list 2-3 key weaknesses]
        Errors: [list specific errors with location, type, description, severity (critical/high/medium/low), suggested fix]
        Suggested Changes: [list specific suggested improvements]
        Overall Assessment: [correct/promising_but_flawed/fundamentally_flawed/incomplete/unclear]
        Confidence in Review: [0.0-1.0]
        """

        self.logger.debug(f"Generating review from {reviewer_id} for {solution['solver_id']}")
        response = self._call_llm(reviewer_id, prompt)
        return self._parse_review(response, reviewer_id, solution["solver_id"])

    def _call_llm(self, llm_name: str, prompt: str) -> str:
        # Use the LLM client
        return self.llm_client.call_llm(llm_name, prompt)

    def _parse_review(self, response: str, reviewer_id: str, solution_id: str) -> PeerReview:
        # Default values
        strengths = ["Clear reasoning", "Good structure"]
        weaknesses = ["Could be more detailed", "Missing some edge cases"]
        errors = []
        suggested_changes = ["Add more explanation", "Include verification steps"]
        overall_assessment = Assessment.PROMISING_BUT_FLAWED
        confidence = 0.7

        try:
            # Clean the response
            response_clean = response.strip()

            # Try to parse as JSON first
            if response_clean.startswith('{'):
                try:
                    data = json.loads(response_clean)

                    strengths = data.get("strengths", strengths)
                    weaknesses = data.get("weaknesses", weaknesses)
                    suggested_changes = data.get("suggested_changes", suggested_changes)
                    confidence = data.get("confidence", confidence)

                    # Parse errors
                    error_data = data.get("errors", [])
                    for err in error_data:
                        try:
                            error = Error(
                                location=err.get("location", ""),
                                error_type=ErrorType(err.get("error_type", "logical_error")),
                                description=err.get("description", ""),
                                severity=Severity(err.get("severity", "medium")),
                                suggested_fix=err.get("suggested_fix", "")
                            )
                            errors.append(error)
                        except:
                            pass

                    # Parse assessment
                    assessment_str = data.get("overall_assessment", "promising_but_flawed")
                    try:
                        overall_assessment = Assessment(assessment_str)
                    except ValueError:
                        overall_assessment = Assessment.PROMISING_BUT_FLAWED
                except json.JSONDecodeError:
                    # Fall back to text parsing if JSON is invalid
                    pass

            # Parse as text format if JSON parsing failed or wasn't attempted
            if not response_clean.startswith('{'):
                lines = response_clean.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("Strengths:"):
                        current_section = "strengths"
                    elif line.startswith("Weaknesses:"):
                        current_section = "weaknesses"
                    elif line.startswith("Errors:"):
                        current_section = "errors"
                    elif line.startswith("Suggested Changes:"):
                        current_section = "suggested_changes"
                    elif line.startswith("Overall Assessment:"):
                        assessment_str = line.replace("Overall Assessment:", "").strip().lower()
                        try:
                            overall_assessment = Assessment(assessment_str)
                        except ValueError:
                            overall_assessment = Assessment.PROMISING_BUT_FLAWED
                    elif line.startswith("Confidence in Review:"):
                        try:
                            confidence = float(line.replace("Confidence in Review:", "").strip())
                        except ValueError:
                            confidence = 0.7
                    elif current_section == "strengths" and line.startswith('-'):
                        strength = line[1:].strip()
                        if strength:
                            strengths.append(strength)
                    elif current_section == "weaknesses" and line.startswith('-'):
                        weakness = line[1:].strip()
                        if weakness:
                            weaknesses.append(weakness)
                    elif current_section == "suggested_changes" and line.startswith('-'):
                        change = line[1:].strip()
                        if change:
                            suggested_changes.append(change)

        except Exception as e:
            self.logger.warning(f"Error parsing review: {e}")

        return PeerReview(
            reviewer_id=reviewer_id,
            solution_id=solution_id,
            strengths=strengths,
            weaknesses=weaknesses,
            errors=errors,
            suggested_changes=suggested_changes,
            overall_assessment=overall_assessment,
            confidence=confidence
        )

    def conduct_peer_review(self, solvers: List[str], solutions: Dict[str, Dict], problem: str) -> Dict[
        str, List[PeerReview]]:
        reviews = {solver: [] for solver in solvers}

        for reviewer in solvers:
            for solution_id, solution in solutions.items():
                if reviewer != solution_id:
                    self.logger.info(f"{reviewer} reviewing {solution_id}")
                    review = self.generate_review(reviewer, solution, problem)
                    reviews[solution_id].append(review)

        return reviews