import json
import logging
import re
from typing import Dict, List, Optional, Any
from models import Solution, PeerReview, RefinedSolution

logger = logging.getLogger(__name__)


class JudgmentSystem:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def make_judgment(
        self,
        judge_instance: str,
        problem: str,
        original_solutions: Dict[str, Solution],
        all_reviews: Dict[str, List[PeerReview]],
        refined_solutions: Dict[str, RefinedSolution],
    ) -> Dict[str, Any]:
        solvers = list(refined_solutions.keys())

        data_package = {
            "problem": problem,
            "allowed_winners": solvers,
            "refined_solutions": {
                k: {
                    "refined_answer": refined_solutions[k].refined_answer,
                    "refined_solution": refined_solutions[k].refined_solution,
                    "confidence": refined_solutions[k].confidence,
                }
                for k in solvers
            },
        }

        prompt = f"""Problem: {problem}

Complete debate data:
{json.dumps(data_package, indent=2)}

You are the final Judge. Select the best refined solution.

You MUST choose the winner from this exact list:
{solvers}

OUTPUT RULES (FOLLOW EXACTLY):
- Return ONLY one JSON object.
- No markdown, no code fences, no extra text before/after JSON.
- Use double quotes for all keys and string values.
- "confidence" must be a number between 0 and 1.
- No extra text.
- reasoning must be <= 2 sentences.
- Keep the entire output under 1200 characters.
Return JSON in this schema:
{{
  "winner": "gemini-2",
  "confidence": 0.0,
  "reasoning": "explain why this is best",
  "evaluation_criteria": {{
    "Logical Soundness": 0,
    "Completeness": 0,
    "Error Handling": 0,
    "Peer Review Integration": 0
  }},
  "ranking": {{
    "gemini-2": 1,
    "gemini-3": 2,
    "gemini-4": 3
  }}
}}
"""

        response = self.llm_client.call_llm(judge_instance, prompt)
        return self._parse_judgment_strict(response, solvers)

    def _strip_code_fences(self, text: str) -> str:
        # remove  ```json ... ```
        t = (text or "").strip()
        t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, flags=re.IGNORECASE)
        return t.strip()

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        t = self._strip_code_fences(text)
        if not t:
            return None


        if t.startswith("{"):
            try:
                return json.loads(t)
            except Exception:
                pass


        first = t.find("{")
        last = t.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = t[first:last + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass


        m = re.search(r"\{.*\"winner\".*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None

    def _coerce_confidence(self, data: Dict[str, Any]) -> None:
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            return
        if isinstance(conf, str):
            s = conf.strip()
            try:
                if s.endswith("%"):
                    val = float(s[:-1].strip()) / 100.0
                else:
                    val = float(s)
                data["confidence"] = val
            except Exception:
                pass

    def _parse_judgment_strict(self, response: str, solvers: List[str]) -> Dict[str, Any]:
        raw = (response or "").strip()
        data = self._try_parse_json(raw)

        if not isinstance(data, dict):
            msg = "Judge output was not valid JSON."
            logger.warning(msg)
            return {
                "winner": None,
                "confidence": None,
                "reasoning": None,
                "evaluation_criteria": None,
                "ranking": None,
                "warning": True,
                "error": msg,
                "raw_judge_output": raw,
                "solvers": solvers,
            }

        self._coerce_confidence(data)

        winner = data.get("winner", None)
        if not winner or not isinstance(winner, str):
            msg = "Judge JSON missing required field: 'winner'."
            logger.warning(msg)
            return {
                "winner": None,
                "confidence": data.get("confidence"),
                "reasoning": data.get("reasoning"),
                "evaluation_criteria": data.get("evaluation_criteria"),
                "ranking": data.get("ranking"),
                "warning": True,
                "error": msg,
                "raw_judge_output": raw,
                "solvers": solvers,
            }

        winner = winner.strip()
        if solvers and winner not in solvers:
            msg = f"Judge chose winner '{winner}', but it is not in solver list {solvers}."
            logger.warning(msg)
            return {
                "winner": None,
                "confidence": data.get("confidence"),
                "reasoning": data.get("reasoning"),
                "evaluation_criteria": data.get("evaluation_criteria"),
                "ranking": data.get("ranking"),
                "warning": True,
                "error": msg,
                "raw_judge_output": raw,
                "solvers": solvers,
            }

        data.setdefault("confidence", None)
        data.setdefault("reasoning", None)
        data.setdefault("evaluation_criteria", None)
        data.setdefault("ranking", None)
        data["warning"] = False
        data["error"] = None
        return data
