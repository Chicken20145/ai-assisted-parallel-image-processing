from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "downloads"
EXTERNAL_DIR = ROOT / "data" / "external"
ARCHIVE = DOWNLOAD_DIR / "BSDS300-images.tgz"
IMAGE_ROOT = EXTERNAL_DIR / "BSDS300" / "images"
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
EXPECTED_SHA256 = "A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(ARCHIVE)

    actual_hash = sha256(ARCHIVE)
    if actual_hash != EXPECTED_SHA256:
        raise SystemExit(f"Invalid BSDS300 checksum: expected {EXPECTED_SHA256}, got {actual_hash}")

    if not IMAGE_ROOT.is_dir():
        safe_extract(ARCHIVE, EXTERNAL_DIR)

    count = sum(1 for _ in IMAGE_ROOT.rglob("*.jpg"))
    if count != 300:
        raise SystemExit(f"BSDS300 must contain 300 JPEG images, found {count}")
    print(f"BSDS300 ready: {count} images at {IMAGE_ROOT}")


if __name__ == "__main__":
    main()
