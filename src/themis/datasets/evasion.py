"""Synthesize short slur and insult samples that evade text filters.

The distilled student inherits the teacher's blind spots on obfuscated abuse
(see :mod:`themis.models.deberta.correction`). This module builds reviewed
``text``/``category`` rows for the bypass patterns a filter-evading author
actually uses, so post-training can correct them: leetspeak, symbol and
homoglyph substitution, zero-width and combining-mark insertion, letter
separators, reversal, stretching, mixed case, fullwidth forms and phonetic
respelling.

Samples stay single-word or very short, because those are the hardest cases:
there is no surrounding sentence for a model to lean on. ``none`` rows are
deliberate hard negatives -- benign words whose letter runs resemble abuse --
so the corrections do not simply teach the model to refuse every substring.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

from themis.models.deberta.dataset import LABELS

# Short slurs already present in the reviewed dataset, one token each.
SLUR_TOKENS: Final = (
    "chink", "kike", "spic", "wetback", "raghead", "faggot", "nigger", "gook", "tranny",
)
# Short, direct insults rather than slurs. Insults need an explicit action.
INSULT_TOKENS: Final = (
    "idiot", "moron", "retard", "loser", "scumbag", "vermin", "freak", "trash",
)
# Words that resemble abuse as substrings or near-misses; must stay "none".
BENIGN_TOKENS: Final = (
    "scunthorpe", "therapist", "assassin", "grasshopper", "classic", "passage",
    "shiitake", "peacock", "raccoon", "titan", "storage", "dragon",
)

_LEET: Final = str.maketrans("aeiostbgz", "431057892")
_SYMBOLS: Final = str.maketrans("aeiosl", "@3!0$1")
# Visually identical Cyrillic/Greek letters, the classic homoglyph bypass.
_HOMOGLYPHS: Final = str.maketrans("aeocpxyik", "аеосрхѵік")

FRAMES: Final = ("{token}", "you {token}", "{token}.", "you're a {token}", "stupid {token}")


def _leet(token: str) -> str:
    return token.translate(_LEET)


def _symbols(token: str) -> str:
    return token.translate(_SYMBOLS)


def _homoglyphs(token: str) -> str:
    return token.translate(_HOMOGLYPHS)


def _fullwidth(token: str) -> str:
    return "".join(chr(ord(char) + 0xFEE0) if char.isascii() and char.isalnum() else char
                   for char in token)


def _zero_width(token: str) -> str:
    return "\u200b".join(token)


def _combining(token: str) -> str:
    return "".join(char + "\u0301" if char in "aeiou" else char for char in token)


def _spaced(token: str) -> str:
    return " ".join(token)


def _dotted(token: str) -> str:
    return ".".join(token)


def _dashed(token: str) -> str:
    return "-".join(token)


def _starred(token: str) -> str:
    return "*".join(token)


def _interleaved(token: str) -> str:
    return "!".join(token)


def _reversed(token: str) -> str:
    return token[::-1]


def _reversed_spaced(token: str) -> str:
    return " ".join(token[::-1])


def _stretched(token: str) -> str:
    return token[:-1] + token[-1] * 3


def _mixed_case(token: str) -> str:
    return "".join(char.upper() if index % 2 else char.lower() for index, char in enumerate(token))


def _underscored(token: str) -> str:
    return token[0] + "_" + token[1:]


def _phonetic(token: str) -> str:
    return token.replace("ck", "k").replace("ph", "f").replace("gg", "g").replace("oo", "u")


# Each pattern maps a bare token to one bypass variant. Order is stable so the
# generated file is reproducible.
TRANSFORMS: Final[tuple[tuple[str, Callable[[str], str]], ...]] = (
    ("plain", lambda token: token),
    ("leet", _leet),
    ("symbols", _symbols),
    ("homoglyphs", _homoglyphs),
    ("fullwidth", _fullwidth),
    ("zero_width", _zero_width),
    ("combining", _combining),
    ("spaced", _spaced),
    ("dotted", _dotted),
    ("dashed", _dashed),
    ("starred", _starred),
    ("interleaved", _interleaved),
    ("reversed", _reversed),
    ("reversed_spaced", _reversed_spaced),
    ("stretched", _stretched),
    ("mixed_case", _mixed_case),
    ("underscored", _underscored),
    ("phonetic", _phonetic),
)
BENIGN_TRANSFORMS: Final = ("plain", "leet", "symbols", "spaced", "dotted", "mixed_case")

_NONE_ACTION: Final = "allow"
_INSULT_ACTION: Final = "block_and_review"


def _record(text: str, category: str, action: str | None = None) -> dict[str, str]:
    if category not in LABELS:
        raise ValueError(f"Unknown category {category!r}")
    record = {"text": text, "category": category}
    if action is not None:
        record["action"] = action
    return record


def _abuse_records() -> Iterable[dict[str, str]]:
    groups = (
        (SLUR_TOKENS, "slur", None),
        (INSULT_TOKENS, "insult", _INSULT_ACTION),
    )
    for tokens, category, action in groups:
        for token_index, token in enumerate(tokens):
            for pattern_index, (_, transform) in enumerate(TRANSFORMS):
                variant = transform(token)
                frame = FRAMES[(token_index + pattern_index) % len(FRAMES)]
                yield _record(frame.format(token=variant), category, action)


def _benign_records() -> Iterable[dict[str, str]]:
    lookup = dict(TRANSFORMS)
    for token_index, token in enumerate(BENIGN_TOKENS):
        for pattern_index, name in enumerate(BENIGN_TRANSFORMS):
            variant = lookup[name](token)
            frame = FRAMES[(token_index + pattern_index) % len(FRAMES)]
            yield _record(frame.format(token=variant), "none", _NONE_ACTION)


def build_evasion_samples() -> list[dict[str, str]]:
    """Return reviewed short abuse rows with filter-evasion and hard negatives.

    Rows are ordered, deduplicated by text, and consistent: a repeated text
    always carries the same category and action. Every category is one of
    ``themis.models.deberta.dataset.LABELS``. Abuse rows use ``insult`` for
    direct attacks and ``slur`` for group-directed ones; ``insult`` rows carry
    an explicit ``block_and_review`` action because it has no default.
    """
    rows: list[dict[str, str]] = []
    seen: dict[str, tuple[str, str | None]] = {}
    for record in (*_abuse_records(), *_benign_records()):
        key = (record["category"], record.get("action"))
        text = record["text"]
        if text in seen:
            if seen[text] != key:
                raise ValueError(f"Conflicting labels for {text!r}")
            continue
        seen[text] = key
        rows.append(record)
    return rows


def write_evasion_samples(path: str | Path) -> Path:
    """Write :func:`build_evasion_samples` rows as UTF-8 JSONL and return the path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as writer:
        for record in build_evasion_samples():
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination
