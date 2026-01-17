from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class LLMConfig:
    name: str
    api_key: str = ""
    endpoint: str = ""
    model: str = ""
    temperature: float = 0.1

@dataclass
class Problem:
    id: int
    category: str
    question: str
    verifiable_answer: Any
    difficulty: str = "high"

LLM_PROVIDERS = {
    "gpt-4": LLMConfig(name="GPT-4", model="gpt-4"),
    "claude": LLMConfig(name="Claude-3", model="claude-3-opus"),
    "gemini": LLMConfig(name="Gemini-Pro", model="gemini-pro"),
    "grok": LLMConfig(name="Grok", model="grok-beta")
}

DATASET_PATH = "dataset/problems.json"
RESULTS_PATH = "results/"
MAX_TOKENS = 4000
SYSTEM_TEMP = 0.1