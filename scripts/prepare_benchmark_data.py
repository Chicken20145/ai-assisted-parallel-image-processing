from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "external" / "BSDS300" / "images"
OUTPUT_ROOT = ROOT / "data" / "external" / "benchmark_suite"
SIZES = [(256, 256), (512, 512), (1920, 1080), (2560, 1440), (3840, 2160)]
SOURCES = {
    "landscape": "train/20008.jpg",
    "detail": "train/159029.jpg",
    "low_contrast": "train/100075.jpg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(source: Image.Image, category: str, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.fit(source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    if category == "low_contrast":
        image = ImageEnhance.Contrast(image).enhance(0.35)
        image = ImageEnhance.Brightness(image).enhance(0.75)
    return image


def main() -> None:
    missing = [str(SOURCE_ROOT / rel) for rel in SOURCES.values() if not (SOURCE_ROOT / rel).is_file()]
    if missing:
        raise SystemExit("Thiếu BSDS300. Chạy scripts/download_datasets.ps1 trước.\n" + "\n".join(missing))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    for category, relative_source in SOURCES.items():
        source_path = SOURCE_ROOT / relative_source
        with Image.open(source_path) as source:
            for width, height in SIZES:
                output_path = OUTPUT_ROOT / f"{category}_{width}x{height}.png"
                prepare_image(source, category, (width, height)).save(output_path, optimize=True)
                records.append(
                    {
                        "file": output_path.name,
                        "category": category,
                        "width": width,
                        "height": height,
                        "source_dataset": "BSDS300",
                        "source_file": relative_source,
                        "transformation": "center_crop_resize_low_contrast" if category == "low_contrast" else "center_crop_resize",
                        "sha256": sha256(output_path),
                    }
                )

    metadata = OUTPUT_ROOT / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Created {len(records)} benchmark images at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
