"""Decision questions for the Perplexity classifier."""

import json


def validate_question(question: object) -> None:
    """Check serialization, leaving native question validation to the model."""
    if not isinstance(question, dict):
        raise ValueError("A question object is required")
    try:
        json.dumps(question, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Question must contain finite JSON values") from exc

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
