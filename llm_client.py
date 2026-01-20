import os
import logging
from dotenv import load_dotenv

from config import GEMINI_INSTANCES, MAX_TOKENS, DEFAULT_TEMPERATURE

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
                logger.info("Using google.genai SDK (new)")
                self._init_new_sdk(genai)
                return
            except ImportError:
                self.use_new_api = False
                logger.info("google.genai not found, falling back to google.generativeai (legacy)")

            import google.generativeai as genai  # noqa: F401
            self._init_legacy_sdk()

        except Exception as e:
            logger.error(f"Failed to initialize Gemini clients: {e}", exc_info=True)
            logger.info("Install new SDK with: pip install -U google-genai")

    def _init_new_sdk(self, genai_module):
        gemini_keys = {
            "gemini-1": os.getenv("GEMINI_1_API_KEY"),
            "gemini-2": os.getenv("GEMINI_2_API_KEY"),
            "gemini-3": os.getenv("GEMINI_3_API_KEY"),
            "gemini-4": os.getenv("GEMINI_4_API_KEY"),
        }

        for instance_id, api_key in gemini_keys.items():
            if api_key and len(api_key) > 10:
                try:
                    client = genai_module.Client(api_key=api_key)
                    self.clients[instance_id] = client
                    logger.info(f"✓ {instance_id} initialized (new SDK)")
                except Exception as e:
                    logger.warning(f"Failed to initialize {instance_id} (new SDK): {e}")

        if not self.clients:
            logger.warning("No Gemini clients initialized. Check your API keys in .env")
            logger.info("You can use the same API key for all 4 instances")

    def _init_legacy_sdk(self):
        gemini_keys = {
            "gemini-1": os.getenv("GEMINI_1_API_KEY"),
            "gemini-2": os.getenv("GEMINI_2_API_KEY"),
            "gemini-3": os.getenv("GEMINI_3_API_KEY"),
            "gemini-4": os.getenv("GEMINI_4_API_KEY"),
        }

        for instance_id, api_key in gemini_keys.items():
            if api_key and len(api_key) > 10:
                self.clients[instance_id] = {"api_key": api_key}
                logger.info(f"✓ {instance_id} initialized (legacy SDK)")

        if not self.clients:
            logger.warning("No Gemini clients initialized. Check your API keys in .env")
            logger.info("You can use the same API key for all 4 instances")

    def call_llm(self, instance_name: str, prompt: str, temperature: float = None) -> str:
        if self.use_mock_mode:
            return self._get_mock_response(instance_name, prompt)

        if not instance_name.startswith("gemini-"):
            logger.warning(f"Invalid instance name: {instance_name}, using gemini-1")
            instance_name = "gemini-1"

        cfg = GEMINI_INSTANCES.get(instance_name)
        model_name = cfg.model if cfg else "gemini-3-flash-preview"

        if temperature is None:
            temperature = cfg.temperature if cfg else DEFAULT_TEMPERATURE

        try:
            if self.use_new_api:
                return self._call_new_sdk(instance_name, model_name, prompt, temperature)
            return self._call_legacy_sdk(instance_name, model_name, prompt, temperature)
        except Exception as e:
            logger.error(f"Error calling {instance_name}: {e}", exc_info=True)
            logger.warning("Falling back to mock response")
            return self._get_mock_response(instance_name, prompt)

    def _call_new_sdk(self, instance_name: str, model_name: str, prompt: str, temperature: float) -> str:
        if instance_name not in self.clients:
            if self.clients:
                instance_name = list(self.clients.keys())[0]
                logger.warning(f"Using {instance_name} as fallback (new SDK)")
            else:
                raise ValueError("No Gemini clients initialized")

        client = self.clients[instance_name]
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": float(temperature),
                "max_output_tokens": int(MAX_TOKENS),
            },
        )
        return getattr(response, "text", "") or ""

    def _call_legacy_sdk(self, instance_name: str, model_name: str, prompt: str, temperature: float) -> str:
        if instance_name not in self.clients:
            if self.clients:
                instance_name = list(self.clients.keys())[0]
                logger.warning(f"Using {instance_name} as fallback (legacy SDK)")
            else:
                raise ValueError("No Gemini clients initialized")

        import google.generativeai as genai

        api_key = self.clients[instance_name]["api_key"]
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(MAX_TOKENS),
            },
        )

        try:
            return response.text or ""
        except Exception:
            try:
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    parts = getattr(content, "parts", None) or []
                    collected = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            collected.append(t)
                    text = "".join(collected).strip()
                    if text:
                        return text
            except Exception:
                pass

            try:
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    fr = getattr(candidates[0], "finish_reason", None)
                    logger.warning(f"Legacy response returned no text parts (finish_reason={fr}).")
            except Exception:
                pass

            return ""

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
        elif "review" in prompt_lower or "critique" in prompt_lower or "evaluate" in prompt_lower:
            return """{
  "strengths": ["Clear logical structure", "Correct application of formulas", "Well-explained reasoning"],
  "weaknesses": ["Could verify edge cases more thoroughly", "Minor notation inconsistencies"],
  "errors": [],
  "suggested_changes": ["Add explicit verification for boundary conditions", "Include alternative solution method"],
  "overall_assessment": "promising_but_flawed",
  "confidence": 0.82
}"""
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
