from typing import List, Dict, Any
import datetime
import json
import re
from models import Solution
from llm_client import LLMClient


class SolutionGenerator:
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.llm_client = LLMClient()
        import logging
        self.logger = logging.getLogger(__name__)

    def generate_solution(self, solver_id: str, problem: str) -> Solution:
        prompt = f"""
        Problem: {problem}

        Generate a complete solution with:
        1. Step-by-step reasoning
        2. Clear assumptions
        3. Final answer
        4. Confidence level (0.0 to 1.0)

        Format:
        Reasoning Steps:
        1. [step 1]
        2. [step 2]
        ...

        Assumptions:
        - [assumption 1]
        - [assumption 2]

        Final Answer: [answer]
        Confidence: [0.0-1.0]
        """

        self.logger.debug(f"Generating solution for {solver_id}")
        response = self._call_llm(solver_id, prompt)
        return self._parse_solution(response, solver_id)

    def _call_llm(self, llm_name: str, prompt: str) -> str:
        # Use the LLM client
        return self.llm_client.call_llm(llm_name, prompt)

    def _parse_solution(self, response: str, solver_id: str) -> Solution:
        # Parse the response to extract structured information
        reasoning_steps = []
        assumptions = []
        final_answer = ""
        confidence = 0.5  # Default confidence

        # Clean up the response
        response_clean = response.strip()

        # Try to extract reasoning steps
        if "Reasoning Steps:" in response_clean:
            if "Assumptions:" in response_clean:
                reasoning_section = response_clean.split("Reasoning Steps:")[1].split("Assumptions:")[0]
            elif "Final Answer:" in response_clean:
                reasoning_section = response_clean.split("Reasoning Steps:")[1].split("Final Answer:")[0]
            else:
                reasoning_section = response_clean.split("Reasoning Steps:")[1]

            lines = reasoning_section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith(
                        ('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.'))):
                    # Remove the number prefix
                    step = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if step:  # Only add non-empty steps
                        reasoning_steps.append(step)

        # Try to extract assumptions
        if "Assumptions:" in response_clean:
            if "Final Answer:" in response_clean:
                assumptions_section = response_clean.split("Assumptions:")[1].split("Final Answer:")[0]
            else:
                assumptions_section = response_clean.split("Assumptions:")[1]

            lines = assumptions_section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-'):
                    assumption = line[1:].strip()
                    if assumption:
                        assumptions.append(assumption)

        # Try to extract final answer
        answer_patterns = [
            r'Final Answer:\s*(.+)',
            r'Answer:\s*(.+)',
            r'Solution:\s*(.+)',
            r'Result:\s*(.+)'
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                # Remove any trailing confidence or other markers
                if 'Confidence:' in final_answer:
                    final_answer = final_answer.split('Confidence:')[0].strip()
                break

        # Try to extract confidence
        confidence_match = re.search(r'Confidence:\s*([0-9.]+)', response_clean, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = 0.5

        # If no reasoning steps were parsed, use the entire response as solution text
        if not reasoning_steps:
            reasoning_steps = [response_clean]

        return Solution(
            solver_id=solver_id,
            solution_text=response_clean,
            final_answer=final_answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            assumptions=assumptions,
            timestamp=datetime.datetime.now().isoformat()
        )

    def generate_all_solutions(self, solvers: List[str], problem: str) -> Dict[str, Solution]:
        solutions = {}
        for solver in solvers:
            self.logger.info(f"Generating solution for {solver}")
            solution = self.generate_solution(solver, problem)
            solutions[solver] = solution
        return solutions