import tempfile
import unittest
from pathlib import Path

import polars as pl

from distill_jev.datasets.download_2b2t_chat import load as load_2b2t
from distill_jev.datasets.download_minecraft_server_chat import load as load_minecraft


class DatasetLoadingTests(unittest.TestCase):
    def test_loaders_return_matching_lazy_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            minecraft = Path(directory) / "minecraft.jsonl"
            csv = Path(directory) / "chats.csv"
            _ = minecraft.write_text(
                '{"username":"player","content":"hello","date":"09/02/2020 2:14 AM","extra":1}\n'
            )
            _ = csv.write_text(
                'time,chat,player_name,player_uuid\n2025-01-01,"hello, world\nsecond line",player,uuid\n'
            )
            first, second = load_minecraft(minecraft), load_2b2t(csv)
            self.assertIsInstance(first, pl.LazyFrame)
            self.assertIsInstance(second, pl.LazyFrame)
            self.assertEqual(first.collect_schema(), second.collect_schema())
            self.assertEqual(first.collect().columns, ["username", "content", "date"])
            self.assertEqual(first.collect().row(0), ("player", "hello", "09/02/2020 2:14 AM"))
            self.assertEqual(second.collect().row(0), ("player", "hello, world\nsecond line", "2025-01-01"))
            self.assertEqual(pl.concat([first, second]).collect().height, 2)

    def test_csv_preserves_empty_and_numeric_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chats.csv"
            _ = path.write_text('time,chat,player_name\n2025,00123,player\n2025,,player\n')
            self.assertEqual(load_2b2t(path).collect()["content"].to_list(), ["00123", ""])
