"""
Configuration for Multi-Gemini Debate System
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class GeminiConfig:
    """Configuration for a Gemini instance"""
    name: str
    instance_id: str
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.1


@dataclass
class Problem:
    """Problem definition"""
    id: int
    category: str
    question: str
    verifiable_answer: Any
    difficulty: str = "high"


# 4 Gemini instances configuration
GEMINI_INSTANCES = {
    "gemini-1": GeminiConfig(
        name="Gemini Instance 1",
        instance_id="gemini-1",
        model="gemini-3-flash-preview"
    ),
    "gemini-2": GeminiConfig(
        name="Gemini Instance 2",
        instance_id="gemini-2",
        model="gemini-3-flash-preview"
    ),
    "gemini-3": GeminiConfig(
        name="Gemini Instance 3",
        instance_id="gemini-3",
        model="gemini-3-flash-preview"
    ),
    "gemini-4": GeminiConfig(
        name="Gemini Instance 4",
        instance_id="gemini-4",
        model="gemini-3-flash-preview"
    ),
}

# Dataset and results paths
DATASET_PATH = "dataset/problems.json"
RESULTS_PATH = "results/"

# Generation parameters
MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.1

# Debate parameters
MIN_SOLVERS = 3
JUDGE_COUNT = 1
TOTAL_INSTANCES = 4
