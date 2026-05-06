import re
from typing import Any


REWARD_NAME = "medical_smoke"
REWARD_TYPE = "batch"


def _extract_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _format_reward(response: str) -> float:
    pattern = re.compile(r".*<thinking>.*?</thinking>\s*<answer>.*?</answer>.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response.strip()) else 0.0


def _lexical_overlap(response: str, ground_truth: str) -> float:
    answer = _extract_answer(response).lower()
    truth = ground_truth.lower()
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer))
    truth_tokens = set(re.findall(r"[a-z0-9]+", truth))
    if not answer_tokens or not truth_tokens:
        return 0.0
    return len(answer_tokens & truth_tokens) / len(truth_tokens)


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    content_weight = 1.0 - format_weight
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input.get("ground_truth", "")
        format_score = _format_reward(response)
        overlap_score = _lexical_overlap(response, ground_truth)
        scores.append(
            {
                "overall": format_weight * format_score + content_weight * overlap_score,
                "format": format_score,
                "accuracy_smoke": overlap_score,
            }
        )
    return scores
