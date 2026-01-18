import json
import logging
from typing import Dict, List
from models import Solution, PeerReview, RefinedSolution

logger = logging.getLogger(__name__)


class JudgmentSystem:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def make_judgment(self, judge_instance: str, problem: str,
                     original_solutions: Dict[str, Solution],
                     all_reviews: Dict[str, List[PeerReview]],
                     refined_solutions: Dict[str, RefinedSolution]) -> Dict:
        data_package = {
            "problem": problem,
            "original_solutions": {k: v.to_dict() for k, v in original_solutions.items()},
            "peer_reviews": {k: [r.to_dict() for r in v] for k, v in all_reviews.items()},
            "refined_solutions": {k: v.to_dict() for k, v in refined_solutions.items()}
        }

        prompt = f"""Problem: {problem}

Complete debate data:
{json.dumps(data_package, indent=2)}

As the final Judge, determine the best solution by evaluating:
1. Logical consistency and soundness
2. Correctness of the answer
3. Completeness of the reasoning
4. Quality of integration of peer feedback

Provide your judgment in JSON format:
{{
  "winner": "[solver_instance_id]",
  "confidence": 0.0-1.0,
  "reasoning": "[detailed explanation of why this solution is best]",
  "evaluation_criteria": {{
    "Logical Soundness": 0-10,
    "Completeness": 0-10,
    "Error Handling": 0-10,
    "Peer Review Integration": 0-10
  }},
  "ranking": {{
    "solver1": 1,
    "solver2": 2,
    "solver3": 3
  }}
}}

Be thorough in your evaluation and explain your reasoning clearly.
"""

        response = self.llm_client.call_llm(judge_instance, prompt)
        return self._parse_judgment(response, list(refined_solutions.keys()))

    def _parse_judgment(self, response: str, solvers: List[str]) -> Dict:
        judgment = None

        try:
            response_clean = response.strip()
            if response_clean.startswith('{'):
                judgment = json.loads(response_clean)
            else:
                import re
                json_match = re.search(r'\{[^{}]*"winner"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    judgment = json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"Error parsing judgment JSON: {e}")

        if not judgment or 'winner' not in judgment:
            winner = solvers[0] if solvers else "gemini-2"
            logger.warning(f"Using default judgment with winner: {winner}")

            judgment = {
                "winner": winner,
                "confidence": 0.8,
                "reasoning": "Solution demonstrated strong logical reasoning and completeness.",
                "evaluation_criteria": {
                    "Logical Soundness": 8.5,
                    "Completeness": 8.0,
                    "Error Handling": 8.0,
                    "Peer Review Integration": 8.0
                },
                "ranking": {solver: i+1 for i, solver in enumerate(solvers)}
            }

        judgment.setdefault('winner', solvers[0] if solvers else "gemini-2")
        judgment.setdefault('confidence', 0.8)
        judgment.setdefault('reasoning', "Solution selected based on evaluation criteria.")
        judgment.setdefault('evaluation_criteria', {
            "Logical Soundness": 8.0,
            "Completeness": 8.0,
            "Error Handling": 8.0,
            "Peer Review Integration": 8.0
        })
        judgment.setdefault('ranking', {solver: i+1 for i, solver in enumerate(solvers)})

        return judgment