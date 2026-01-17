import json
import re
from typing import Any, Dict, List


def parse_json_response(response: str) -> Dict[str, Any]:
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except json.JSONDecodeError:
        return {}


def validate_response_structure(response: Dict[str, Any], required_keys: List[str]) -> bool:
    return all(key in response for key in required_keys)


def extract_final_answer(text: str) -> str:
    patterns = [
        r'Final Answer:\s*(.+)',
        r'Answer:\s*(.+)',
        r'Solution:\s*(.+)',
        r'Result:\s*(.+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def calculate_agreement_score(reviews: List[Dict]) -> float:
    if not reviews:
        return 0.0

    positive_terms = ["correct", "strong", "valid", "accurate", "complete"]
    negative_terms = ["wrong", "error", "flaw", "incomplete", "incorrect"]

    scores = []
    for review in reviews:
        text = json.dumps(review).lower()
        pos_count = sum(term in text for term in positive_terms)
        neg_count = sum(term in text for term in negative_terms)

        if pos_count + neg_count > 0:
            score = pos_count / (pos_count + neg_count)
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0