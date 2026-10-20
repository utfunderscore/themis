import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from themis.datasets.evasion import (
    BENIGN_TOKENS,
    INSULT_TOKENS,
    SLUR_TOKENS,
    build_evasion_samples,
    write_evasion_samples,
)
from themis.models.deberta.correction import obfuscation_key
from themis.models.deberta.dataset import ACTION_LABELS, LABELS


class EvasionSamplesTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_evasion_samples()

    def test_every_row_has_a_known_category_and_valid_action(self):
        for row in self.rows:
            self.assertIsInstance(row["text"], str)
            self.assertTrue(row["text"].strip())
            self.assertIn(row["category"], LABELS)
            if "action" in row:
                self.assertIn(row["action"], ACTION_LABELS)
        # Insult has no default action, so those rows must carry one.
        for row in self.rows:
            if row["category"] == "insult":
                self.assertEqual(row["action"], "block_and_review")

    def test_texts_are_unique(self):
        texts = [row["text"] for row in self.rows]
        self.assertEqual(len(texts), len(set(texts)))

    def test_abuse_and_hard_negatives_are_both_present(self):
        by_category = {}
        for row in self.rows:
            by_category.setdefault(row["category"], []).append(row["text"])
        self.assertTrue({"slur", "insult", "none"} <= set(by_category))
        self.assertTrue(any(text == token for token in SLUR_TOKENS
                            for text in by_category["slur"]))
        self.assertTrue(any(token in text for token in BENIGN_TOKENS
                            for text in by_category["none"]))
        self.assertTrue(any(token in text for token in INSULT_TOKENS
                            for text in by_category["insult"]))

    def test_bypass_patterns_are_covered(self):
        abuse = {row["text"] for row in self.rows if row["category"] != "none"}
        # Plain, leet, symbols, separators, zero-width and reversal on a base token.
        self.assertIn("chink", abuse)
        self.assertIn("you ch1nk", abuse)
        self.assertIn("c h i n k.", abuse)
        self.assertIn("c\u200bh\u200bi\u200bn\u200bk", abuse)
        self.assertIn("knihc.", abuse)
        self.assertTrue(any("$" in text or "@" in text for text in abuse))
        # Homoglyph and fullwidth variants are byte-distinct from the plain token.
        self.assertIn("you're a сhіnк", abuse)
        self.assertIn("stupid ｃｈｉｎｋ", abuse)

    def test_benign_variants_stay_none(self):
        for row in self.rows:
            if row["category"] == "none":
                self.assertEqual(row.get("action"), "allow")
        keys = {obfuscation_key(text) for text in
                (row["text"] for row in self.rows if row["category"] == "none")}
        self.assertTrue({"scunthorpe", "therapist", "grasshopper"} <= keys)

    def test_write_is_deterministic_and_unicode_preserving(self):
        with TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.jsonl", Path(temp) / "b.jsonl"
            write_evasion_samples(first)
            write_evasion_samples(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded, self.rows)
            self.assertIn("\u200b", first.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
