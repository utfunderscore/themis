from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import polars as pl

from distill_jev.datasets.download_2b2t_chat import DATASET_URL, download


def make_archive(files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    buffer.seek(0)
    return buffer


class Download2b2tChatTests(unittest.TestCase):
    def test_extracts_only_nested_chats_and_removes_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache" / "chats.csv"
            response = make_archive({"data/chats.csv": b"message\nhello\n", "other.csv": b"other"})
            with patch("distill_jev.datasets.download_2b2t_chat.urlopen", return_value=response) as request:
                self.assertIsInstance(download(output), pl.LazyFrame)
                http_request = request.call_args.args[0]
                self.assertEqual(http_request.full_url, DATASET_URL)
                self.assertEqual(http_request.get_header("User-agent"), "distill-jev/0.1 (dataset downloader)")
            self.assertEqual(output.read_bytes(), b"message\nhello\n")
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_reuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chats.csv"
            output.write_bytes(b"cached")
            with patch("distill_jev.datasets.download_2b2t_chat.urlopen") as request:
                self.assertIsInstance(download(output), pl.LazyFrame)
                request.assert_not_called()

    def test_force_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chats.csv"
            output.write_bytes(b"old")
            with patch(
                "distill_jev.datasets.download_2b2t_chat.urlopen",
                return_value=make_archive({"chats.csv": b"new"}),
            ):
                download(output, force=True)
            self.assertEqual(output.read_bytes(), b"new")

    def test_missing_or_ambiguous_csv_preserves_existing_file(self) -> None:
        for files in ({"other.csv": b"other"}, {"a/chats.csv": b"a", "b/chats.csv": b"b"}):
            with self.subTest(files=files), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "chats.csv"
                output.write_bytes(b"old")
                with patch(
                    "distill_jev.datasets.download_2b2t_chat.urlopen",
                    return_value=make_archive(files),
                ):
                    with self.assertRaises(ValueError):
                        download(output, force=True)
                self.assertEqual(output.read_bytes(), b"old")
                self.assertEqual(list(output.parent.iterdir()), [output])
