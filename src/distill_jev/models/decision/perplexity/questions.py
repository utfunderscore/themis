"""Decision questions for the Perplexity classifier."""

import json
from typing import cast


def validate_question(question: object) -> None:
    """Validate native question structure, independently of RPC row limits.

    Match the existing predictor's permissive criteria-description semantics;
    the pinned native renderer remains responsible for rendering descriptions.
    """
    if not isinstance(question, dict):
        raise ValueError("A question object is required")
    question = cast(dict[str, object], question)
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(cast(dict[object, object], criteria)) <= 255:
            raise ValueError("Choice questions require 1–255 named criteria")
        if any(not isinstance(key, str) for key in cast(dict[object, object], criteria)):
            raise ValueError("Choice criterion names must be strings")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(cast(list[object], criteria)) <= 255:
            raise ValueError("Score questions require 2–255 ordered criteria")
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(cast(dict[object, object], criteria)) - {"false", "true"}
        ):
            raise ValueError("Noul criteria may only describe false and true")
    else:
        raise ValueError("Question type must be choice, score, or noul")
    _ = json.dumps(question, allow_nan=False)

CATEGORY_QUESTION = {
    "type": "choice",
    "instructions": "Which single type of offensive content is this message, if any?",
    "criteria": {
        "none": "Not offensive.",
        "profanity": (
            "Strong swearing that is excessive or crude enough to act on, "
            "not ordinary casual swearing."
        ),
        "insult": "Personal insults or harassment aimed at a player.",
        "slur": "Racial, homophobic, or other identity-based slurs or hate speech.",
        "sexual": "Sexual or explicit content.",
        "threat": (
            "Threats of violence, self-harm encouragement, doxxing. "
            "The game may involve violence like killing over players, dealing "
            "damage, or punching, but actions violence outside of the game "
            "is not allowed."
        ),
    },
}
