"""Resume official per-file downloads when Kaggle's ZIP endpoint returns 404."""

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
LOCAL = threading.local()
RATE_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def pace_request():
    global NEXT_REQUEST
    with RATE_LOCK:
        now = time.monotonic()
        delay = max(0, NEXT_REQUEST - now)
        NEXT_REQUEST = max(now, NEXT_REQUEST) + 0.25
    if delay:
        time.sleep(delay)


def api():
    if not hasattr(LOCAL, "api"):
        LOCAL.api = KaggleApi()
        LOCAL.api.authenticate()
    return LOCAL.api


def fetch(root, name):
    destination = root / name
    if destination.exists() and destination.stat().st_size > 0:
        return
    for attempt in range(10):
        try:
            pace_request()
            api().competition_download_file(COMPETITION, name, path=str(root), quiet=True, force=True)
            if not destination.exists() or destination.stat().st_size == 0:
                raise RuntimeError(f"Missing download: {name}")
            return
        except Exception as error:
            response = getattr(error, "response", None)
            if response is not None and response.status_code == 429:
                global NEXT_REQUEST
                wait = max(120, int(response.headers.get("Retry-After", "120")))
                with RATE_LOCK:
                    NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + wait)
                print(f"Rate limit: pausing all requests for at least {wait}s", flush=True)
            if attempt == 9 or (response is not None and response.status_code in (401, 403, 404)):
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scope", choices=("all", "train", "test"), default="all")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in ("train_labels.csv", "test.csv", "sample_submission.csv"):
        fetch(args.root, name)
    names = []
    for csv_name, folder in (("train_labels.csv", "train_images"), ("test.csv", "test_images")):
        if args.scope != "all" and not folder.startswith(args.scope):
            continue
        with (args.root / csv_name).open() as stream:
            for row in csv.DictReader(stream):
                name = row["filename"]
                if Path(name).name != name:
                    raise ValueError(f"Unexpected filename in official manifest: {name}")
                names.append(f"{folder}/{name}")
    pending = [name for name in names if not (args.root / name).exists()]
    start = time.monotonic()
    print(f"Official manifest: {len(names)} images; {len(pending)} still to download", flush=True)
    # Only the eight standard API workers below access the official per-file endpoint.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for count, _ in enumerate(executor.map(lambda name: fetch(args.root, name), pending), 1):
            if count % 200 == 0 or count == len(pending):
                elapsed = time.monotonic() - start
                print(f"Downloaded {count}/{len(pending)} new files; {count/elapsed:.1f} files/s", flush=True)
    result = {"competition": COMPETITION, "scope": args.scope, "image_count": len(names), "method": "official Kaggle SDK individual files", "elapsed_seconds": time.monotonic()-start}
    (args.root / f"download_receipt_{args.scope}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
