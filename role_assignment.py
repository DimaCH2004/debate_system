import json
import re
from typing import Dict, List, Tuple, Any
from models import RolePreference
from llm_client import LLMClient


class RoleAssignment:
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.llm_client = LLMClient()
        import logging
        self.logger = logging.getLogger(__name__)

    def get_role_preference(self, llm_name: str, problem: str) -> RolePreference:
        prompt = f"""
        Problem: {problem}

        You will participate in a debate system with three Solvers and one Judge.
        Based on the problem above, assess your capabilities:

        1. Which role would you prefer? (Solver or Judge)
        2. Rate your confidence for each role (0.0 to 1.0)
        3. Provide reasoning for your preference

        Format your response as:
        Preferred roles (in order): [list]
        Confidence by role: {{"Solver": 0.0, "Judge": 0.0}}
        Reasoning: [explanation]
        Self-assessment: [brief assessment of your strengths/weaknesses for this problem]
        """

        self.logger.debug(f"Getting role preference for {llm_name}")
        response = self._call_llm(llm_name, prompt)
        return self._parse_role_preference(response, llm_name)

    def _call_llm(self, llm_name: str, prompt: str) -> str:
        # Use the LLM client
        return self.llm_client.call_llm(llm_name, prompt)

    def _parse_role_preference(self, response: str, llm_name: str) -> RolePreference:
        # Default values
        preferred_roles = ["Solver"]
        confidence_by_role = {"Solver": 0.8, "Judge": 0.5}
        reasoning = f"I prefer to be a Solver for this problem."
        self_assessment = "I can analyze mathematical problems effectively."

        # Try to parse the response
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Preferred roles"):
                    try:
                        # Extract list from: Preferred roles (in order): ["Solver", "Judge"]
                        match = re.search(r'\[(.*?)\]', line)
                        if match:
                            roles_str = match.group(1)
                            preferred_roles = [role.strip().strip('"\'') for role in roles_str.split(',')]
                    except:
                        pass
                elif line.startswith("Confidence by role"):
                    try:
                        # Extract dict from: Confidence by role: {"Solver": 0.85, "Judge": 0.65}
                        match = re.search(r'\{.*\}', line)
                        if match:
                            dict_str = match.group(0)
                            confidence_by_role = json.loads(dict_str)
                    except:
                        pass
                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
                elif line.startswith("Self-assessment:"):
                    self_assessment = line.replace("Self-assessment:", "").strip()
        except Exception as e:
            self.logger.warning(f"Error parsing role preference for {llm_name}: {e}")

        return RolePreference(
            preferred_roles=preferred_roles,
            confidence_by_role=confidence_by_role,
            reasoning=reasoning,
            self_assessment=self_assessment
        )

    def assign_roles_deterministic(self, preferences: Dict[str, RolePreference]) -> Dict[str, str]:
        llm_names = list(preferences.keys())

        if len(llm_names) < 4:
            raise ValueError(f"Need at least 4 LLMs, but only have {len(llm_names)}")

        judge_candidates = []
        solver_candidates = []

        for name, pref in preferences.items():
            if "Judge" in pref.preferred_roles and pref.confidence_by_role.get("Judge", 0) > 0.7:
                judge_candidates.append((name, pref.confidence_by_role.get("Judge", 0)))
            else:
                solver_score = pref.confidence_by_role.get("Solver", 0)
                solver_candidates.append((name, solver_score))

        self.logger.debug(f"Judge candidates: {judge_candidates}")
        self.logger.debug(f"Solver candidates: {solver_candidates}")

        if len(judge_candidates) > 0:
            judge_candidates.sort(key=lambda x: x[1], reverse=True)
            judge = judge_candidates[0][0]
            solver_names = [name for name in llm_names if name != judge]

            if len(solver_names) > 3:
                solver_candidates = [(n, preferences[n].confidence_by_role.get("Solver", 0))
                                     for n in solver_names]
                solver_candidates.sort(key=lambda x: x[1], reverse=True)
                solver_names = [n for n, _ in solver_candidates[:3]]
        else:
            all_candidates = [(name, preferences[name].confidence_by_role.get("Solver", 0))
                              for name in llm_names]
            all_candidates.sort(key=lambda x: x[1])
            judge = all_candidates[0][0]
            solver_names = [name for name, _ in all_candidates[1:]]

        if len(solver_names) > 3:
            solver_names = solver_names[:3]

        return {
            "judge": judge,
            "solvers": solver_names
        }