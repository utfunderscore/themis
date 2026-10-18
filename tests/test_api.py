import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from themis.api import create_app


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, texts, batch_size=32):
        self.calls.append((texts, batch_size))
        return [dict(category=text, action="allow", category_probabilities={text: 1.0},
                     action_probabilities={"allow": 1.0}) for text in texts]


class ApiTests(unittest.TestCase):
    def test_batch_preserves_order_and_duplicates(self):
        predictor = FakePredictor()
        with TestClient(create_app(predictor=predictor, batch_size=2)) as client:
            response = client.post("/v1/inference", json=["hello", "bye", "hello"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["category"] for item in response.json()], ["hello", "bye", "hello"])
        self.assertEqual(predictor.calls, [(["hello", "bye", "hello"], 2)])
        self.assertEqual(response.json()[0]["category_probabilities"], {"hello": 1.0})

    def test_validation_and_empty_batch(self):
        predictor = FakePredictor()
        with TestClient(create_app(predictor=predictor)) as client:
            self.assertEqual(client.post("/v1/inference", json=[]).json(), [])
            for payload in ({"chats": ["hello"]}, [123], [None], [""], ["a" * 16385], ["a"] * 257):
                with self.subTest(payload=str(payload)[:80]):
                    self.assertEqual(client.post("/v1/inference", json=payload).status_code, 422)
        self.assertEqual(predictor.calls, [])

    def test_openapi_file_matches_app(self):
        schema = json.loads((Path(__file__).parents[1] / "openapi.json").read_text())
        self.assertEqual(create_app().openapi(), schema)

    def test_wrong_result_count_fails(self):
        class BrokenPredictor:
            def predict(self, texts, batch_size=32):
                return []

        with TestClient(create_app(predictor=BrokenPredictor()), raise_server_exceptions=False) as client:
            self.assertEqual(client.post("/v1/inference", json=["hello"]).status_code, 500)
