"""Verify and import the two official folders downloaded with Safari."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    receipt = {
        "source": "https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data",
        "method": "Safari: select each train_images/test_images folder, then Download",
        "external_dataset_used": False,
        "folders": {},
    }
    # Validate the entire manifest before copying; keep the user's downloads intact.
    for folder, manifest, expected_count in (
        ("train_images", "train_labels.csv", 50000),
        ("test_images", "test.csv", 2000),
    ):
        source = args.downloads / folder
        destination = args.root / folder
        with (args.root / manifest).open(newline="") as stream:
            records = list(csv.DictReader(stream))
        names = {row["filename"] for row in records}
        found = {p.name for p in source.iterdir() if p.is_file()}
        assert len(records) == len(names) == expected_count
        assert names == found, {"missing": sorted(names-found)[:10], "extra": sorted(found-names)[:10]}
        digest, total_bytes, matched_existing = hashlib.sha256(), 0, 0
        for name in sorted(names):
            assert Path(name).name == name
            path = source / name
            raw = path.read_bytes()
            digest.update(name.encode() + hashlib.sha256(raw).digest())
            total_bytes += len(raw)
            with Image.open(path) as image:
                image.load()
                assert image.size == (32, 32) and image.mode == "RGB", str(path)
            existing = destination / name
            if existing.exists():
                assert existing.read_bytes() == raw, f"Official API/browser mismatch: {name}"
                matched_existing += 1
        receipt["folders"][folder] = {
            "images": expected_count, "bytes": total_bytes,
            "ordered_sha256": digest.hexdigest(), "existing_api_files_matched": matched_existing,
            "all_images_decoded": True, "all_names_match_official_csv": True,
        }
    for folder in receipt["folders"]:
        source, destination = args.downloads / folder, args.root / folder
        destination.mkdir(exist_ok=True)
        for path in source.iterdir():
            target = destination / path.name
            if not target.exists():
                shutil.copy2(path, target)
    receipt["verified_at"] = datetime.now().astimezone().isoformat()
    (args.root / "official_import_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
