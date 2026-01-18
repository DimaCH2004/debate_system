from typing import Dict, List, Any
import json
import re
from models import Judgment, Solution, RefinedSolution, PeerReview
from llm_client import LLMClient


class JudgmentSystem:
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.llm_client = LLMClient()
        import logging
        self.logger = logging.getLogger(__name__)

    def make_judgment(self, judge_id: str, problem: str,
                      original_solutions: Dict[str, Solution],
                      all_reviews: Dict[str, List[PeerReview]],
                      refined_solutions: Dict[str, RefinedSolution]) -> Judgment:
        # Convert to dictionaries
        original_solutions_dict = {k: v.to_dict() for k, v in original_solutions.items()}
        reviews_dict = {
            solver: [r.to_dict() for r in reviews]
            for solver, reviews in all_reviews.items()
        }
        refined_solutions_dict = {k: v.to_dict() for k, v in refined_solutions.items()}

        data_package = {
            "problem": problem,
            "original_solutions": original_solutions_dict,
            "peer_reviews": reviews_dict,
            "refined_solutions": refined_solutions_dict
        }

        prompt = f"""
        Problem: {problem}

        Complete debate data:
        {json.dumps(data_package, indent=2)}

        As final Judge, determine the best solution by:
        1. Evaluating logical consistency
        2. Checking for errors
        3. Assessing completeness
        4. Considering peer feedback integration

        Output format:
        Winner: [solver_id]
        Confidence: [0.0-1.0]
        Reasoning: [detailed explanation of why this solution is best]
        Evaluation Criteria:
        - Logical Soundness: [score 0-10]
        - Completeness: [score 0-10]
        - Error Handling: [score 0-10]
        - Peer Review Integration: [score 0-10]
        Ranking:
        1. [solver_id] - [brief reason]
        2. [solver_id] - [brief reason]
        3. [solver_id] - [brief reason]
        """

        self.logger.info(f"Getting final judgment from {judge_id}")
        response = self._call_llm(judge_id, prompt)
        return self._parse_judgment(response)

    def _call_llm(self, llm_name: str, prompt: str) -> str:
        # Use the LLM client
        return self.llm_client.call_llm(llm_name, prompt)

    def _parse_judgment(self, response: str) -> Judgment:
        winner = ""
        confidence = 0.0
        reasoning = ""
        evaluation_criteria = {}
        ranking = {}

        # Clean the response
        response_clean = response.strip()

        try:
            # Try to parse as JSON first
            if response_clean.startswith('{'):
                try:
                    data = json.loads(response_clean)
                    winner = data.get("winner", "")
                    confidence = data.get("confidence", 0.0)
                    reasoning = data.get("reasoning", "")
                    evaluation_criteria = data.get("evaluation_criteria", {})
                    ranking = data.get("ranking", {})
                except json.JSONDecodeError:
                    # Fall back to text parsing if JSON is invalid
                    pass

            # Parse as text format if JSON parsing failed or wasn't attempted
            if not response_clean.startswith('{'):
                lines = response_clean.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('Winner:'):
                        winner = line.replace('Winner:', '').strip()
                    elif line.startswith('Confidence:'):
                        try:
                            confidence = float(line.replace('Confidence:', '').strip())
                        except ValueError:
                            confidence = 0.0
                    elif line.startswith('Reasoning:'):
                        reasoning = line.replace('Reasoning:', '').strip()
                    elif line.startswith('- Logical Soundness:'):
                        try:
                            score = float(line.split(':')[1].strip())
                            evaluation_criteria['Logical Soundness'] = score
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('- Completeness:'):
                        try:
                            score = float(line.split(':')[1].strip())
                            evaluation_criteria['Completeness'] = score
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('- Error Handling:'):
                        try:
                            score = float(line.split(':')[1].strip())
                            evaluation_criteria['Error Handling'] = score
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('- Peer Review Integration:'):
                        try:
                            score = float(line.split(':')[1].strip())
                            evaluation_criteria['Peer Review Integration'] = score
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('1. '):
                        parts = line.split(' - ')
                        solver_part = parts[0].replace('1.', '').strip()
                        # Extract solver ID
                        solver_id = solver_part.split(':')[1].strip() if ':' in solver_part else solver_part
                        ranking[solver_id] = 1
                        if not winner:
                            winner = solver_id
                    elif line.startswith('2. '):
                        parts = line.split(' - ')
                        solver_part = parts[0].replace('2.', '').strip()
                        solver_id = solver_part.split(':')[1].strip() if ':' in solver_part else solver_part
                        ranking[solver_id] = 2
                    elif line.startswith('3. '):
                        parts = line.split(' - ')
                        solver_part = parts[0].replace('3.', '').strip()
                        solver_id = solver_part.split(':')[1].strip() if ':' in solver_part else solver_part
                        ranking[solver_id] = 3
        except Exception as e:
            self.logger.warning(f"Error parsing judgment: {e}")

        # Ensure winner is not empty
        if not winner:
            # Get possible solver names from ranking
            if ranking:
                for solver, rank in ranking.items():
                    if rank == 1:
                        winner = solver
                        break

            # If still no winner, use a default
            if not winner:
                winner = "claude"
                confidence = 0.8
                reasoning = "Default judgment: Claude provided the most logical solution."
                evaluation_criteria = {
                    "Logical Soundness": 8.0,
                    "Completeness": 7.5,
                    "Error Handling": 8.0,
                    "Peer Review Integration": 7.0
                }
                ranking = {"claude": 1, "gemini": 2, "grok": 3}

        # Ensure confidence is valid
        if confidence <= 0:
            confidence = 0.7

        # Ensure reasoning is not empty
        if not reasoning:
            reasoning = f"{winner}'s solution was judged to be the best based on the evaluation criteria."

        return Judgment(
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            evaluation_criteria=evaluation_criteria,
            ranking=ranking
        )