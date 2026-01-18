import re
import logging
from typing import List, Dict
from models import Solution

logger = logging.getLogger(__name__)


class SolutionGenerator:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_solution(self, solver_id: str, problem: str) -> Solution:
        prompt = f"""Problem: {problem}

Provide a complete solution with:
1. Step-by-step reasoning
2. Clear assumptions
3. Final answer
4. Confidence level (0.0 to 1.0)

Format:
Reasoning Steps:
1. [step]
2. [step]
...

Assumptions:
- [assumption]
- [assumption]

Final Answer: [your answer]
Confidence: [0.0-1.0]
"""

        response = self.llm_client.call_llm(solver_id, prompt)
        return self._parse_solution(response, solver_id)

    def _parse_solution(self, response: str, solver_id: str) -> Solution:
        reasoning_steps = []
        assumptions = []
        final_answer = ""
        confidence = 0.5

        try:
            if "Reasoning Steps:" in response:
                start = response.index("Reasoning Steps:") + len("Reasoning Steps:")
                end = response.index("Assumptions:") if "Assumptions:" in response else len(response)
                steps_text = response[start:end]

                for line in steps_text.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        step = re.sub(r'^\d+[\.\)]\s*', '', line)
                        step = step.lstrip('- ')
                        if step:
                            reasoning_steps.append(step)
            if "Assumptions:" in response:
                start = response.index("Assumptions:") + len("Assumptions:")
                end = response.index("Final Answer:") if "Final Answer:" in response else len(response)
                assump_text = response[start:end]

                for line in assump_text.split('\n'):
                    line = line.strip()
                    if line.startswith('-'):
                        assumptions.append(line[1:].strip())

            for pattern in [r'Final Answer:\s*(.+)', r'Answer:\s*(.+)']:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                    if 'Confidence:' in final_answer:
                        final_answer = final_answer.split('Confidence:')[0].strip()
                    break

            match = re.search(r'Confidence:\s*([0-9.]+)', response)
            if match:
                confidence = float(match.group(1))

            if not reasoning_steps:
                reasoning_steps = [response]

        except Exception as e:
            logger.warning(f"Error parsing solution: {e}")
            reasoning_steps = [response]

        return Solution(
            solver_id=solver_id,
            solution_text=response,
            final_answer=final_answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            assumptions=assumptions
        )

    def generate_all_solutions(self, solvers: List[str], problem: str) -> Dict[str, Solution]:
        solutions = {}
        for solver in solvers:
            logger.info(f"  Generating solution from {solver}")
            solutions[solver] = self.generate_solution(solver, problem)
        return solutions