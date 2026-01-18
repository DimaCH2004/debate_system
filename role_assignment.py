import json
import re
import logging
from typing import Dict
from models import RolePreference

logger = logging.getLogger(__name__)


class RoleAssignment:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def get_role_preference(self, llm_name: str, problem: str) -> RolePreference:
        prompt = f"""Problem: {problem}

You will participate in a debate system with three Solvers and one Judge.

Assess your capabilities for this problem:
1. Which role would you prefer? (Solver or Judge)
2. Rate your confidence for each role (0.0 to 1.0)
3. Provide reasoning for your preference

Format:
Preferred roles (in order): ["Solver", "Judge"]
Confidence by role: {{"Solver": 0.85, "Judge": 0.70}}
Reasoning: [your reasoning]
Self-assessment: [your strengths/weaknesses for this problem]
"""

        response = self.llm_client.call_llm(llm_name, prompt)
        return self._parse_preference(response)

    def _parse_preference(self, response: str) -> RolePreference:
        # Defaults
        preferred_roles = ["Solver"]
        confidence_by_role = {"Solver": 0.8, "Judge": 0.5}
        reasoning = "Default preference"
        self_assessment = "General capabilities"

        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()

                if "Preferred roles" in line:
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        roles_str = match.group(1)
                        preferred_roles = [r.strip().strip('"\'')
                                           for r in roles_str.split(',')]

                elif "Confidence by role" in line:
                    match = re.search(r'\{.*?\}', line)
                    if match:
                        try:
                            confidence_by_role = json.loads(match.group(0))
                        except:
                            pass

                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()

                elif line.startswith("Self-assessment:"):
                    self_assessment = line.replace("Self-assessment:", "").strip()

        except Exception as e:
            logger.warning(f"Error parsing preference: {e}")

        return RolePreference(
            preferred_roles=preferred_roles,
            confidence_by_role=confidence_by_role,
            reasoning=reasoning,
            self_assessment=self_assessment
        )

    def assign_roles(self, preferences: Dict[str, RolePreference]) -> Dict:
        llm_names = list(preferences.keys())

        if len(llm_names) < 4:
            raise ValueError(f"Need 4 LLMs, have {len(llm_names)}")

        # Find best judge candidate
        judge_candidates = []
        for name, pref in preferences.items():
            if "Judge" in pref.preferred_roles:
                score = pref.confidence_by_role.get("Judge", 0)
                judge_candidates.append((name, score))

        if judge_candidates:
            judge_candidates.sort(key=lambda x: x[1], reverse=True)
            judge = judge_candidates[0][0]
        else:
            solver_scores = [(n, preferences[n].confidence_by_role.get("Solver", 0))
                             for n in llm_names]
            solver_scores.sort(key=lambda x: x[1])
            judge = solver_scores[0][0]

        solvers = [n for n in llm_names if n != judge][:3]

        return {"judge": judge, "solvers": solvers}