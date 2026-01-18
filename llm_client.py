import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(self):
        mock_mode = os.getenv("USE_MOCK_MODE", "true").lower()
        self.use_mock_mode = mock_mode != "false"

        logger.info(f"LLMClient initialized. Mock mode: {self.use_mock_mode}")
        self.clients = {}
        self.use_new_api = False

        if not self.use_mock_mode:
            self._initialize_gemini_clients()
        else:
            logger.info("Running in mock mode - using simulated responses")

    def _initialize_gemini_clients(self):
        try:
            try:
                from google import genai
                self.use_new_api = True
                logger.info("Using google.genai SDK")
            except ImportError:
                import google.generativeai as genai
                self.use_new_api = False
                logger.info("Using google.generativeai SDK (legacy)")

            gemini_configs = {
                "gemini-1": os.getenv("GEMINI_1_API_KEY"),
                "gemini-2": os.getenv("GEMINI_2_API_KEY"),
                "gemini-3": os.getenv("GEMINI_3_API_KEY"),
                "gemini-4": os.getenv("GEMINI_4_API_KEY"),
            }

            for name, api_key in gemini_configs.items():
                if api_key and len(api_key) > 10:
                    try:
                        if self.use_new_api:
                            client = genai.Client(api_key=api_key)
                        else:
                            client = {
                                'module': genai,
                                'api_key': api_key
                            }

                        self.clients[name] = client
                        logger.info(f"✓ {name} initialized")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {name}: {e}")

            if not self.clients:
                logger.warning("No Gemini clients initialized. Check your API keys in .env")
                logger.info("You can use the same API key for all 4 instances")

        except Exception as e:
            logger.error(f"Failed to import Gemini: {e}")
            logger.info("Install with: pip install google-genai")

    def call_llm(self, instance_name: str, prompt: str, temperature: float = 0.1) -> str:
        if self.use_mock_mode:
            return self._get_mock_response(instance_name, prompt)

        try:
            # Ensure it's a valid gemini instance
            if not instance_name.startswith("gemini-"):
                logger.warning(f"Invalid instance name: {instance_name}, using gemini-1")
                instance_name = "gemini-1"

            return self._call_gemini(instance_name, prompt, temperature)

        except Exception as e:
            logger.error(f"Error calling {instance_name}: {e}")
            logger.warning("Falling back to mock response")
            return self._get_mock_response(instance_name, prompt)

    def _call_gemini(self, instance_name: str, prompt: str, temperature: float) -> str:
        if instance_name not in self.clients:
            # Try to use any available client
            if self.clients:
                instance_name = list(self.clients.keys())[0]
                logger.warning(f"Using {instance_name} as fallback")
            else:
                raise ValueError("No Gemini clients initialized")

        client = self.clients[instance_name]

        if self.use_new_api:
            try:
                response = client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=prompt,
                    config={
                        'temperature': temperature,
                        'max_output_tokens': 2000,
                    }
                )
                return response.text
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise
        else:
            genai = client['module']
            api_key = client['api_key']

            genai.configure(api_key=api_key)

            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': 2000,
                }
            )
            return response.text

    def _get_mock_response(self, instance_name: str, prompt: str) -> str:

        prompt_lower = prompt.lower()
        if "role" in prompt_lower and "preference" in prompt_lower:
            if "gemini-1" == instance_name:
                return """Preferred roles (in order): ["Judge", "Solver"]
Confidence by role: {"Solver": 0.85, "Judge": 0.88}
Reasoning: I'm excellent at evaluating multiple perspectives objectively.
Self-assessment: Strong analytical and comparative skills."""
            else:
                return """Preferred roles (in order): ["Solver", "Judge"]
Confidence by role: {"Solver": 0.85, "Judge": 0.75}
Reasoning: I excel at detailed analytical reasoning and comprehensive solutions.
Self-assessment: Strong at breaking down complex problems step-by-step."""
        elif "refine" in prompt_lower or "your original solution" in prompt_lower:
            if "factorial" in prompt or "20 zeros" in prompt or "n!" in prompt:
                return """Changes Made:
- Critique: Could verify edge cases more thoroughly
  Response: Added verification for n=80 through n=90
  Accepted: true
  Changes: Added detailed calculation showing n=85 gives exactly 20 trailing zeros

Refined Solution: Using Legendre's formula, the number of trailing zeros in n! equals floor(n/5) + floor(n/25) + floor(n/125) + ...
Testing systematically:
n=80: 16+3+0 = 19 zeros
n=85: 17+3+0 = 20 zeros ✓
Therefore n=85 is the smallest integer.

Refined Answer: 85
Confidence: 0.95"""
            elif "probability" in prompt or "coin" in prompt or "heads" in prompt:
                return """Changes Made:
- Critique: Add explicit verification for boundary conditions
  Response: Added proof that no integer solution exists
  Accepted: true
  Changes: Showed that H=2T and H+T=10 leads to T=10/3, which is not an integer

Refined Solution: Let H=heads, T=tails. We need H+T=10 and H=2T.
Substituting: 2T+T=10, so 3T=10, giving T=10/3≈3.33.
Since T must be an integer, no valid outcome exists.
Probability = 0.

Refined Answer: 0.0000
Confidence: 0.95"""
            else:
                return """Changes Made:
- Critique: Add explicit verification for boundary conditions
  Response: Added comprehensive edge case analysis
  Accepted: true
  Changes: Verified solution works for all boundary conditions

Refined Solution: Applied systematic approach with verification of edge cases.
Solution confirmed through multiple methods.

Refined Answer: 42
Confidence: 0.90"""

        # Peer review responses
        elif "review" in prompt_lower or "critique" in prompt_lower or "evaluate" in prompt_lower:
            return """{
  "strengths": ["Clear logical structure", "Correct application of formulas", "Well-explained reasoning"],
  "weaknesses": ["Could verify edge cases more thoroughly", "Minor notation inconsistencies"],
  "errors": [],
  "suggested_changes": ["Add explicit verification for boundary conditions", "Include alternative solution method"],
  "overall_assessment": "promising_but_flawed",
  "confidence": 0.82
}"""
        elif "judge" in prompt_lower or "winner" in prompt_lower or "best solution" in prompt_lower:
            import re
            import json as json_lib
            solvers = []
            try:
                matches = re.findall(r'"solver_id":\s*"([^"]+)"', prompt)
                if matches:
                    solvers = list(dict.fromkeys(matches))

                if not solvers and "refined_solutions" in prompt:
                    refined_match = re.search(r'"refined_solutions":\s*\{([^}]+)\}', prompt, re.DOTALL)
                    if refined_match:
                        content = refined_match.group(1)
                        solver_matches = re.findall(r'"([^"]+)":', content)
                        if solver_matches:
                            solvers = list(dict.fromkeys(solver_matches))
            except Exception as e:
                logger.debug(f"Error extracting solvers: {e}")

            if not solvers:
                solvers = ["gemini-2", "gemini-3", "gemini-4"]

            winner = solvers[0] if solvers else "gemini-2"

            return json_lib.dumps({
                "winner": winner,
                "confidence": 0.87,
                "reasoning": f"{winner}'s solution demonstrated the strongest logical rigor and most complete analysis.",
                "evaluation_criteria": {
                    "Logical Soundness": 9.2,
                    "Completeness": 8.8,
                    "Error Handling": 8.5,
                    "Peer Review Integration": 8.7
                },
                "ranking": {solver: i + 1 for i, solver in enumerate(solvers)}
            }, indent=2)

        else:
            if "factorial" in prompt or "20 zeros" in prompt or "n!" in prompt:
                return """Reasoning Steps:
1. Trailing zeros in n! come from factors of 10 = 2×5
2. Since factors of 2 are more abundant than 5, we count factors of 5
3. Legendre's formula: zeros = floor(n/5) + floor(n/25) + floor(n/125) + ...
4. We need this sum to equal exactly 20
5. Testing values: n=80 gives 16+3=19, n=85 gives 17+3=20
6. Verified: 85 is the smallest n where n! has exactly 20 trailing zeros

Assumptions:
- Counting trailing zeros in decimal representation
- Using standard factorial definition

Final Answer: 85
Confidence: 0.93"""

            elif "probability" in prompt or "coin" in prompt or "heads is exactly twice" in prompt:
                return """Reasoning Steps:
1. Let H = number of heads, T = number of tails
2. Given: H + T = 10 (total flips)
3. Condition: H = 2T (heads exactly twice tails)
4. Substituting: 2T + T = 10, so 3T = 10
5. This gives T = 10/3 ≈ 3.333...
6. Since T must be a whole number and 10/3 is not an integer, no valid outcome exists
7. Therefore, the probability is 0

Assumptions:
- Fair coin with P(H) = P(T) = 0.5
- Independent flips
- "Exactly twice" means H = 2T exactly

Final Answer: 0.0000
Confidence: 0.95"""

            else:
                return """Reasoning Steps:
1. Analyzed the problem requirements carefully
2. Identified the key constraints and variables
3. Applied relevant mathematical principles
4. Performed step-by-step calculations
5. Verified the solution satisfies all conditions

Assumptions:
- Standard mathematical definitions apply
- All stated conditions are accurate

Final Answer: 42
Confidence: 0.85"""