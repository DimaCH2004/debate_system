import json
import re
import logging
from typing import Dict, List
from models import PeerReview, Solution, Assessment

logger = logging.getLogger(__name__)


class PeerReviewSystem:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_review(self, reviewer_id: str, solution: Solution,
                        problem: str) -> PeerReview:
        prompt = f"""Problem: {problem}

Solution to review from {solution.solver_id}:
{json.dumps(solution.to_dict(), indent=2)}

Provide structured critique:

Strengths: [list 2-3 key strengths]
Weaknesses: [list 2-3 weaknesses]
Errors: [list specific errors if any]
Suggested Changes: [list improvements]
Overall Assessment: [correct/promising_but_flawed/fundamentally_flawed/incomplete]
Confidence in Review: [0.0-1.0]
"""

        response = self.llm_client.call_llm(reviewer_id, prompt)
        return self._parse_review(response, reviewer_id, solution.solver_id)

    def _parse_review(self, response: str, reviewer_id: str,
                      solution_id: str) -> PeerReview:
        strengths = []
        weaknesses = []
        errors = []
        suggested_changes = []
        overall_assessment = Assessment.PROMISING_BUT_FLAWED
        confidence = 0.7

        try:
            # Try JSON first
            if response.strip().startswith('{'):
                data = json.loads(response.strip())
                strengths = data.get("strengths", strengths)
                weaknesses = data.get("weaknesses", weaknesses)
                suggested_changes = data.get("suggested_changes", suggested_changes)
                confidence = data.get("confidence", confidence)

                assessment_str = data.get("overall_assessment", "promising_but_flawed")
                try:
                    overall_assessment = Assessment(assessment_str)
                except:
                    pass
            else:
                current_section = None
                for line in response.split('\n'):
                    line = line.strip()

                    if line.startswith("Strengths:"):
                        current_section = "strengths"
                    elif line.startswith("Weaknesses:"):
                        current_section = "weaknesses"
                    elif line.startswith("Suggested Changes:"):
                        current_section = "changes"
                    elif line.startswith("Overall Assessment:"):
                        assessment = line.replace("Overall Assessment:", "").strip().lower()
                        try:
                            overall_assessment = Assessment(assessment)
                        except:
                            pass
                    elif line.startswith("Confidence"):
                        try:
                            confidence = float(re.search(r'[0-9.]+', line).group())
                        except:
                            pass
                    elif line.startswith('-') or line.startswith('•'):
                        item = line[1:].strip()
                        if current_section == "strengths":
                            strengths.append(item)
                        elif current_section == "weaknesses":
                            weaknesses.append(item)
                        elif current_section == "changes":
                            suggested_changes.append(item)

        except Exception as e:
            logger.warning(f"Error parsing review: {e}")

        return PeerReview(
            reviewer_id=reviewer_id,
            solution_id=solution_id,
            strengths=strengths or ["Clear reasoning"],
            weaknesses=weaknesses or ["Could be more detailed"],
            errors=errors,
            suggested_changes=suggested_changes or ["Add verification"],
            overall_assessment=overall_assessment,
            confidence=confidence
        )

    def conduct_peer_review(self, solvers: List[str],
                            solutions: Dict[str, Solution],
                            problem: str) -> Dict[str, List[PeerReview]]:
        reviews = {solver: [] for solver in solvers}

        for reviewer in solvers:
            for solution_id, solution in solutions.items():
                if reviewer != solution_id:
                    logger.info(f"  {reviewer} reviewing {solution_id}")
                    review = self.generate_review(reviewer, solution, problem)
                    reviews[solution_id].append(review)

        return reviews