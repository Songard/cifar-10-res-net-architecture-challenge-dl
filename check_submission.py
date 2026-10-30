"""Independently re-read a prediction CSV and its audit sidecar before any upload."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        table = list(csv.reader(stream))
    with (args.data / "sample_submission.csv").open(newline="") as stream:
        expected_ids = [row["id"] for row in csv.DictReader(stream)]
    header, body = table[0], table[1:]
    assert header == ["id", "label"], f"header is {header}"
    assert [row[0] for row in body] == expected_ids, "IDs differ from sample_submission.csv (order or membership)"
    assert all(len(row) == 2 and row[1].isdigit() and 0 <= int(row[1]) <= 9 for row in body), "labels must be integers 0-9"
    audit = json.loads(args.csv.with_suffix(".audit.json").read_text())
    assert audit["sha256"] == sha256(args.csv), "CSV changed after it was generated"
    assert audit["checkpoint_sha256"] == sha256(audit["checkpoint"]), "checkpoint changed after prediction"
    assert audit["parameter_count"] <= 5_000_000
    receipts = Path("artifacts/kaggle_submissions.jsonl")
    uploads = [json.loads(line) for line in receipts.read_text().splitlines() if line.strip()] if receipts.exists() else []
    # The audit sidecar is written before any upload; Kaggle receipts are appended separately after one succeeds.
    kaggle = [{k: r[k] for k in ("submission_ref", "status", "public_score", "private_score")}
              for r in uploads if r["csv_sha256"] == audit["sha256"]]
    counts = [0] * 10
    for row in body:
        counts[int(row[1])] += 1
    print(json.dumps({"csv": str(args.csv), "rows": len(body), "ids_match_sample_submission": True, "labels_valid": True,
                      "csv_sha256": audit["sha256"], "checkpoint": audit["checkpoint"], "arch": audit["arch"],
                      "parameter_count": audit["parameter_count"], "weights": audit["weights"], "tta": audit["tta"],
                      "predicted_class_counts": counts, "kaggle_receipts_for_this_file": kaggle}, indent=2))


if __name__ == "__main__":
    main()
