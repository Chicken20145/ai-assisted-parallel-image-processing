from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "downloads"
EXTERNAL_DIR = ROOT / "data" / "external"
ARCHIVE = DOWNLOAD_DIR / "BSDS300-images.tgz"
IMAGE_ROOT = EXTERNAL_DIR / "BSDS300" / "images"
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
EXPECTED_SHA256 = "A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4"
MAX_DOWNLOAD_ATTEMPTS = 3


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
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
        if sys.version_info >= (3, 12):
            bundle.extractall(destination, filter="data")
        else:
            bundle.extractall(destination)


def download_verified_archive() -> None:
    if ARCHIVE.is_file():
        actual_hash = sha256(ARCHIVE)
        if actual_hash == EXPECTED_SHA256:
            return
        print(f"Removing invalid cached BSDS300 archive ({actual_hash}).")
        ARCHIVE.unlink()

    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        temporary.unlink(missing_ok=True)
        try:
            print(f"Downloading BSDS300 (attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS})...")
            request = urllib.request.Request(URL, headers={"User-Agent": "parallel-image-processing/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)

            actual_hash = sha256(temporary)
            if actual_hash != EXPECTED_SHA256:
                raise RuntimeError(
                    f"checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
                )
            temporary.replace(ARCHIVE)
            return
        except Exception as error:  # Network and checksum failures share the retry path.
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    raise SystemExit(
        f"Unable to download a verified BSDS300 archive after {MAX_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def ensure_extracted_dataset() -> int:
    current_count = sum(1 for _ in IMAGE_ROOT.rglob("*.jpg")) if IMAGE_ROOT.is_dir() else 0
    if current_count == 300:
        return current_count

    dataset_root = EXTERNAL_DIR / "BSDS300"
    extraction_root = EXTERNAL_DIR / ".BSDS300.extracting"
    if current_count:
        print(f"Rebuilding incomplete BSDS300 extraction ({current_count}/300 images).")
    shutil.rmtree(extraction_root, ignore_errors=True)
    extraction_root.mkdir(parents=True)
    try:
        safe_extract(ARCHIVE, extraction_root)
        candidate = extraction_root / "BSDS300"
        candidate_image_root = candidate / "images"
        candidate_count = (
            sum(1 for _ in candidate_image_root.rglob("*.jpg"))
            if candidate_image_root.is_dir()
            else 0
        )
        if candidate_count != 300:
            raise SystemExit(f"BSDS300 must contain 300 JPEG images, found {candidate_count}")

        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        candidate.replace(dataset_root)
        return candidate_count
    finally:
        shutil.rmtree(extraction_root, ignore_errors=True)


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    download_verified_archive()
    count = ensure_extracted_dataset()
    print(f"BSDS300 ready: {count} images at {IMAGE_ROOT}")


if __name__ == "__main__":
    main()
