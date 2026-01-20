import re
import logging
from typing import Dict, List
from models import RefinedSolution, CritiqueResponse, PeerReview, Solution
from utils import extract_final_answer

logger = logging.getLogger(__name__)


class RefinementSystem:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def refine_solution(
        self,
        solver_id: str,
        original_solution: Solution,
        peer_reviews: List[PeerReview],
        problem: str
    ) -> RefinedSolution:
        """
        Fix: Force 'Refined Answer' to appear at the TOP of the response so truncation
        won't remove it. Also make the response shorter to reduce token pressure.
        """
        reviews_summary = "\n\n".join([
            f"Review from {r.reviewer_id}:\n"
            f"- Strengths: {', '.join(r.strengths)}\n"
            f"- Weaknesses: {', '.join(r.weaknesses)}\n"
            f"- Suggestions: {', '.join(r.suggested_changes)}\n"
            f"- Assessment: {r.overall_assessment.value}"
            for r in peer_reviews
        ])

        prompt = f"""Problem: {problem}

Your original solution:
{original_solution.solution_text}

Peer reviews:
{reviews_summary}

IMPORTANT OUTPUT RULES (FOLLOW EXACTLY):
1) Put the final answer at the TOP on its own line as:
   Refined Answer: <answer>
2) Put confidence right after as:
   Confidence: <0.0-1.0>
3) Keep the refined solution concise (do NOT write a long essay).
   Max ~20 lines for the refined solution.
4) Then include changes made in bullet format.

Required format:

Refined Answer: <answer>
Confidence: <0.0-1.0>

Refined Solution:
<concise improved reasoning; focus on correctness; keep short>

Changes Made:
- Critique: <summary>
  Response: <your response>
  Accepted: <true/false>
  Changes: <what you changed>
"""

        response = self.llm_client.call_llm(solver_id, prompt)
        return self._parse_refinement(response, solver_id)

    def _parse_refinement(self, response: str, solver_id: str) -> RefinedSolution:
        """
        Fix: Much stronger answer extraction.
        - Prefer 'Refined Answer:' / 'Final Answer:' labels
        - Fallback: utils.extract_final_answer()
        - Fallback: last plausible number in the response
        """
        changes_made: List[CritiqueResponse] = []
        refined_solution = ""
        refined_answer = ""
        confidence = 0.8

        try:
            text = (response or "").strip()

            # 1) Primary extraction: explicit labels
            answer_found = False
            for pattern in [
                r'^\s*Refined Answer:\s*([^\n\r]+)',
                r'^\s*Final Answer:\s*([^\n\r]+)',
                r'^\s*Answer:\s*([^\n\r]+)',
            ]:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    answer_text = match.group(1).strip()
                    # Strip anything after confidence if the model merges lines
                    if 'confidence' in answer_text.lower():
                        answer_text = re.split(r'confidence\s*:', answer_text, flags=re.IGNORECASE)[0].strip()
                    refined_answer = answer_text
                    answer_found = True
                    break

            # 2) Confidence extraction
            conf_match = re.search(r'^\s*Confidence:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE | re.MULTILINE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except Exception:
                    pass

            # 3) Extract refined solution block
            if "Refined Solution:" in text:
                start = text.index("Refined Solution:") + len("Refined Solution:")
                end = len(text)
                for marker in ["Changes Made:", "Refined Answer:", "Final Answer:", "Confidence:"]:
                    if marker in text[start:]:
                        marker_pos = text.index(marker, start)
                        if marker_pos < end:
                            end = marker_pos
                refined_solution = text[start:end].strip()
            else:
                refined_solution = text

            # 4) Parse Changes Made section (best-effort)
            if "Changes Made:" in text:
                changes_section = text.split("Changes Made:", 1)[1]
                # don't include extra trailing stuff
                if "Refined Solution:" in changes_section:
                    changes_section = changes_section.split("Refined Solution:", 1)[0]

                current_critique = ""
                current_response = ""
                current_accepted = False
                current_changes = ""

                for line in changes_section.splitlines():
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
                        current_response = ""
                        current_accepted = False
                        current_changes = ""
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

            # 5) Fallback answer extraction if empty
            if not answer_found or not refined_answer:
                fallback = extract_final_answer(text)
                if fallback:
                    refined_answer = fallback.strip()
                    answer_found = True

            # 6) Last-resort: grab the last plausible number (handles "= 685,464" cases)
            if not refined_answer:
                # numbers like 685,464 or 685464 or 0.0000 etc.
                candidates = re.findall(r'(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)', text)
                if candidates:
                    refined_answer = candidates[-1].strip()

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

    def refine_all_solutions(
        self,
        solutions: Dict[str, Solution],
        reviews: Dict[str, List[PeerReview]],
        problem: str
    ) -> Dict[str, RefinedSolution]:
        refined = {}
        for solver_id, solution in solutions.items():
            logger.info(f"  Refining solution from {solver_id}")
            refined[solver_id] = self.refine_solution(
                solver_id, solution, reviews[solver_id], problem
            )
        return refined
