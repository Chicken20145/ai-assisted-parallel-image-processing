from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "download_datasets.py"


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_datasets_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for index in range(300):
            content = f"jpeg-{index}".encode()
            member = tarfile.TarInfo(f"BSDS300/images/train/{index:06d}.jpg")
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))


class DatasetDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.module = load_downloader()
        self.module.DOWNLOAD_DIR = self.root / "downloads"
        self.module.EXTERNAL_DIR = self.root / "external"
        self.module.ARCHIVE = self.module.DOWNLOAD_DIR / "BSDS300-images.tgz"
        self.module.IMAGE_ROOT = self.module.EXTERNAL_DIR / "BSDS300" / "images"
        self.module.DOWNLOAD_DIR.mkdir(parents=True)
        self.module.EXTERNAL_DIR.mkdir(parents=True)

        self.source_archive = self.root / "source.tgz"
        create_archive(self.source_archive)
        self.module.EXPECTED_SHA256 = self.module.sha256(self.source_archive)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_invalid_cache_is_replaced_after_retry(self) -> None:
        self.module.ARCHIVE.write_bytes(b"invalid cached response")
        valid_bytes = self.source_archive.read_bytes()
        responses = [io.BytesIO(b"invalid download"), io.BytesIO(valid_bytes)]

        with mock.patch.object(self.module.urllib.request, "urlopen", side_effect=responses) as urlopen:
            with mock.patch.object(self.module.time, "sleep"):
                self.module.download_verified_archive()

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(self.module.sha256(self.module.ARCHIVE), self.module.EXPECTED_SHA256)
        self.assertFalse(Path(f"{self.module.ARCHIVE}.part").exists())

    def test_incomplete_extraction_is_rebuilt_atomically(self) -> None:
        self.module.ARCHIVE.write_bytes(self.source_archive.read_bytes())
        self.module.IMAGE_ROOT.mkdir(parents=True)
        (self.module.IMAGE_ROOT / "old.jpg").write_bytes(b"old")

        count = self.module.ensure_extracted_dataset()

        self.assertEqual(count, 300)
        self.assertEqual(len(list(self.module.IMAGE_ROOT.rglob("*.jpg"))), 300)
        self.assertFalse((self.module.IMAGE_ROOT / "old.jpg").exists())
        self.assertFalse((self.module.EXTERNAL_DIR / ".BSDS300.extracting").exists())


if __name__ == "__main__":
    unittest.main()
