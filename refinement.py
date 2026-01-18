import re
import logging
from typing import Dict, List
from models import RefinedSolution, CritiqueResponse, PeerReview, Solution

logger = logging.getLogger(__name__)


class RefinementSystem:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def refine_solution(self, solver_id: str, original_solution: Solution,
                        peer_reviews: List[PeerReview], problem: str) -> RefinedSolution:
        reviews_summary = "\n\n".join([
            f"Review from {r.reviewer_id}:\n" +
            f"Strengths: {', '.join(r.strengths)}\n" +
            f"Weaknesses: {', '.join(r.weaknesses)}\n" +
            f"Suggestions: {', '.join(r.suggested_changes)}\n" +
            f"Assessment: {r.overall_assessment.value}"
            for r in peer_reviews
        ])

        prompt = f"""Problem: {problem}

Your original solution:
{original_solution.solution_text}

Peer reviews:
{reviews_summary}

Refine your solution by:
1. Addressing each critique explicitly
2. Defending your reasoning if critiques are wrong
3. Incorporating valid feedback

Format:
Changes Made:
- Critique: [summary]
  Response: [your response]
  Accepted: [true/false]
  Changes: [what you changed]

Refined Solution: [complete refined solution]
Refined Answer: [final answer]
Confidence: [0.0-1.0]
"""

        response = self.llm_client.call_llm(solver_id, prompt)
        return self._parse_refinement(response, solver_id)

    def _parse_refinement(self, response: str, solver_id: str) -> RefinedSolution:
        changes_made = []
        refined_solution = ""
        refined_answer = ""
        confidence = 0.8

        try:
            answer_found = False
            for pattern in [r'Refined Answer:\s*([^\n]+)', r'Final Answer:\s*([^\n]+)']:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer_text = match.group(1).strip()
                    if 'Confidence:' in answer_text:
                        answer_text = answer_text.split('Confidence:')[0].strip()

                    refined_answer = answer_text
                    answer_found = True
                    logger.debug(f"Extracted refined answer: '{refined_answer}'")
                    break

            if not answer_found:
                logger.warning("No refined answer found in response")

            match = re.search(r'Confidence:\s*([0-9.]+)', response)
            if match:
                confidence = float(match.group(1))

            if "Refined Solution:" in response:
                start = response.index("Refined Solution:") + len("Refined Solution:")
                end = len(response)
                for marker in ["Refined Answer:", "Confidence:"]:
                    if marker in response[start:]:
                        marker_pos = response.index(marker, start)
                        if marker_pos < end:
                            end = marker_pos
                refined_solution = response[start:end].strip()
            else:
                refined_solution = response

            if "Changes Made:" in response:
                changes_section = response.split("Changes Made:")[1]
                if "Refined Solution:" in changes_section:
                    changes_section = changes_section.split("Refined Solution:")[0]

                current_critique = ""
                current_response = ""
                current_accepted = False
                current_changes = ""

                for line in changes_section.split('\n'):
                    line = line.strip()
                    if line.startswith("- Critique:"):
                        if current_critique:
                            changes_made.append(CritiqueResponse(
                                critique_id=f"critique_{len(changes_made) + 1}",
                                accepted=current_accepted,
                                response=current_response,
                                changes_made=current_changes
                            ))
                        current_critique = line.replace("- Critique:", "").strip()
                    elif line.startswith("Response:"):
                        current_response = line.replace("Response:", "").strip()
                    elif line.startswith("Accepted:"):
                        current_accepted = "true" in line.lower()
                    elif line.startswith("Changes:"):
                        current_changes = line.replace("Changes:", "").strip()

                if current_critique:
                    changes_made.append(CritiqueResponse(
                        critique_id=f"critique_{len(changes_made) + 1}",
                        accepted=current_accepted,
                        response=current_response,
                        changes_made=current_changes
                    ))

            if not changes_made:
                changes_made.append(CritiqueResponse(
                    critique_id="default",
                    accepted=True,
                    response="Incorporated peer feedback",
                    changes_made="Refined solution based on reviews"
                ))

        except Exception as e:
            logger.warning(f"Error parsing refinement: {e}")
            logger.exception(e)

        logger.info(f"Parsed refinement for {solver_id}: answer='{refined_answer}', confidence={confidence}")

        return RefinedSolution(
            original_solution_id=solver_id,
            changes_made=changes_made,
            refined_solution=refined_solution,
            refined_answer=refined_answer,
            confidence=confidence
        )

    def refine_all_solutions(self, solutions: Dict[str, Solution],
                             reviews: Dict[str, List[PeerReview]],
                             problem: str) -> Dict[str, RefinedSolution]:
        refined = {}
        for solver_id, solution in solutions.items():
            logger.info(f"  Refining solution from {solver_id}")
            refined[solver_id] = self.refine_solution(
                solver_id, solution, reviews[solver_id], problem
            )
        return refined