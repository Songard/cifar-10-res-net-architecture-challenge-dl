"""Upload newly delivered candidate CSVs to the Kaggle competition and record one receipt per upload.

Each candidate is a directory holding submission.csv and submission.audit.json, as delivered by the HPC work in
../project1-hpc/artifacts/candidates. A candidate whose CSV sha256 already has a receipt in
artifacts/kaggle_submissions.jsonl is skipped. Before an upload the CSV is checked independently of the
process that produced it: header, row count, id order against data/sample_submission.csv, label range, sha256
against the audit, held-out validation and the parameter limit. The hidden-test predictions are only hashed and
compared row-for-row; nothing else is read from them.

A candidate trained on all 50,000 training images has no held-out accuracy of its own. It is accepted when its
audit or config.json records the full training set, and its generalization estimate is its twin's: the same
recipe trained on the 45k split, named by the audit's "twin", by --twin FULL=TWIN, or by dropping "-full" from
the candidate's name. The twin's held-out accuracy orders the candidates, drives the minimum-accuracy filter and
the public prediction, and is recorded in the receipt next to the fit accuracy on the trained-on split. A
full-training candidate is a final-submission deliverable, so it is uploaded as soon as one submission is left.

The daily quota is counted from the submissions Kaggle itself lists for the current UTC day, and a reserve is
kept so a better candidate delivered later the same day can still be uploaded; only a candidate whose held-out
accuracy beats every receipted one, or a full-training candidate, takes the reserve. Public scores are recorded
with the pre-registered prediction, which uses the transfer slope measured earlier in this project
(validation +0.1 pp ~ public +0.2 pp, anchored at validation 97.4 ~ public 88.6, band +/- 2.0 pp).
"""

import argparse
import csv
import fcntl
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
DAILY_QUOTA = 5
PARAMETER_LIMIT = 5_000_000
FULL_TRAIN_IMAGES = 50_000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def refresh_kaggle_token():
    """Renew the OAuth access token with the stored refresh token.

    kagglesdk (2.2.5) only renews a token 30 minutes *after* it expired, so a run inside that window would
    otherwise fail with "Authentication required"; this does what the CLI does by itself after the window."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials
    with KaggleApi().build_kaggle_client() as client:
        creds = KaggleCredentials.load(client=client)
        if creds:
            creds.refresh_access_token()


def kaggle_submissions(kaggle_bin, retry=True):
    """Return Kaggle's own list of this account's submissions, newest first, parsed from the CLI's CSV output."""
    result = subprocess.run([str(kaggle_bin), "competitions", "submissions", "-c", COMPETITION, "--csv"],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0 and retry and "Authentication required" in (result.stderr + result.stdout):
        refresh_kaggle_token()
        return kaggle_submissions(kaggle_bin, retry=False)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle submissions failed: {result.stderr.strip() or result.stdout.strip()}")
    rows = []
    for row in csv.DictReader(line for line in result.stdout.splitlines() if line.strip()):
        score = row.get("publicScore") or ""
        rows.append({"ref": int(row["ref"]), "date": row["date"], "status": row["status"].split(".")[-1],
                     "description": row.get("description", ""),
                     "public_score": float(score) if score.replace(".", "", 1).isdigit() else None})
    return rows


def candidate_config(directory):
    """The candidate's training config: config.json at the top level, else the one inside its runs/ copy."""
    top = directory / "config.json"
    if top.exists():
        return read_json(top)
    nested = sorted(directory.glob("runs/*/config.json"))
    return read_json(nested[0]) if nested else {}


def full_training(directory, audit):
    """True when the candidate's own records say it was trained on every training image."""
    config = candidate_config(directory)
    images = audit.get("train_images", config.get("train_images"))
    return bool(audit.get("full_train", config.get("full_train"))) or images == FULL_TRAIN_IMAGES


def find_twin(directory, audit, twins):
    """Return (twin name, twin held-out accuracy) for a full-training candidate; (None, None) when unknown.

    --twin FULL=TWIN names the twin directory; --twin FULL=0.9766 is a manual estimate for when the twin
    has not finished training yet (the receipt then marks the twin as a manual estimate)."""
    given, manual = twins.get(directory.name), None
    if given is not None:
        try:
            manual, given = float(given), None
        except ValueError:
            pass
    name = audit.get("twin") or given
    if not name and "-full" in directory.name and (directory.parent / directory.name.replace("-full", "")).is_dir():
        name = directory.name.replace("-full", "")
    accuracy = audit.get("twin_heldout_valid_accuracy")
    if accuracy is None and name:
        twin_audit = read_json(directory.parent / name / "submission.audit.json")
        if twin_audit.get("validation_held_out"):
            accuracy = twin_audit.get("valid_accuracy_of_these_weights")
    if accuracy is None and manual is not None:
        accuracy, name = manual, f"{name or 'unknown'} [manual estimate]"
    return name, (float(accuracy) if accuracy is not None else None)


def check_candidate(directory, sample_ids, twins):
    """Check one candidate; return its CSV digest, audit, problems and what the quota logic needs to know."""
    audit = read_json(directory / "submission.audit.json")
    raw = (directory / "submission.csv").read_bytes()
    rows = list(csv.reader(raw.decode().splitlines()))
    body = rows[1:]
    problems = []
    if rows[0] != ["id", "label"]:
        problems.append(f"header {rows[0]}")
    if len(body) != len(sample_ids):
        problems.append(f"{len(body)} rows, expected {len(sample_ids)}")
    if [r[0] for r in body] != sample_ids:
        problems.append("ids differ from sample_submission.csv")
    if not all(len(r) == 2 and r[1].isdigit() and 0 <= int(r[1]) <= 9 for r in body):
        problems.append("labels outside 0-9")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != audit.get("sha256"):
        problems.append("csv sha256 differs from audit")
    if int(audit.get("parameter_count", PARAMETER_LIMIT + 1)) > PARAMETER_LIMIT:
        problems.append(f"parameter_count {audit.get('parameter_count')} over limit")
    # The accuracy every decision below uses: the candidate's own held-out accuracy, or its twin's for a
    # full-training candidate, whose own validation split was trained on.
    info = {"full": False, "twin": None, "proxy": 0.0}
    if audit.get("validation_held_out", False):
        info["proxy"] = float(audit.get("valid_accuracy_of_these_weights") or 0)
    elif full_training(directory, audit):
        info["full"] = True
        info["twin"], accuracy = find_twin(directory, audit, twins)
        if accuracy is None:
            problems.append(f"full-training candidate without a twin held-out accuracy (pass --twin {directory.name}=TWIN or =0.97xx)")
        else:
            info["proxy"] = accuracy
    else:
        problems.append("validation not held out")
    return digest, audit, problems, info


def describe(directory, audit, info):
    seed = candidate_config(directory).get("seed")
    valid = 100 * info["proxy"]
    prediction = round(88.6 + (valid - 97.4) * 2.0, 1)
    seed_text = f" seed{seed}" if seed is not None else ""
    basis = f"50k full train, twin {info['twin']} held-out {valid:.2f}" if info["full"] else f"45k, val {valid:.2f}"
    text = (f"HPC {directory.name}: {audit.get('arch')} {audit.get('epoch')}ep{seed_text}, {basis}. "
            f"Pred public {prediction} +/- 2.0")
    return text, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(HERE.parent / "project1-hpc" / "artifacts" / "candidates"))
    parser.add_argument("--receipts", default=str(HERE / "artifacts" / "kaggle_submissions.jsonl"))
    parser.add_argument("--kaggle", default=str(HERE / ".venv" / "bin" / "kaggle"))
    parser.add_argument("--max-uploads", type=int, default=3, help="uploads per run")
    parser.add_argument("--reserve", type=int, default=1,
                        help="daily submissions kept for a candidate better than every receipted one")
    parser.add_argument("--min-valid", type=float, default=None,
                        help="skip candidates whose held-out accuracy is below this (default: best receipted minus 0.0015)")
    parser.add_argument("--full-margin", type=float, default=0.001,
                        help="how far below --min-valid a full-training candidate's twin may be and still upload")
    parser.add_argument("--twin", nargs="*", default=[], metavar="FULL=TWIN",
                        help="twin (45k) candidate of a full-training candidate whose audit does not name one; "
                             "FULL=0.9766 gives a manual held-out estimate while the twin is still training")
    parser.add_argument("--force", nargs="*", default=[], help="candidate names to upload regardless of --min-valid")
    parser.add_argument("--margin", type=float, default=0.001,
                        help="candidates within this of the best receipted held-out accuracy get first call on the quota")
    parser.add_argument("--keep-for-better", type=int, default=2,
                        help="submissions a weaker candidate must leave unused (beyond --reserve) for a stronger one")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    twins = dict(pair.split("=", 1) for pair in args.twin)

    sample_ids = [r[0] for r in list(csv.reader((HERE / "data" / "sample_submission.csv").read_text().splitlines()))[1:]]
    receipts_path = Path(args.receipts)
    # Two runs at once (the scheduled check and a watcher) must not both upload the same candidate:
    # hold an exclusive lock for the whole run, and read the receipts only after getting it.
    lock = open(receipts_path.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    receipts = [json.loads(line) for line in receipts_path.read_text().splitlines() if line.strip()]
    known = {r["csv_sha256"] for r in receipts}
    # Quota is scarce, so by default only candidates close to the best already uploaded are worth a public score;
    # anything weaker is skipped unless named in --force (a different architecture, say). Full-training receipts
    # carry no held-out accuracy of their own and so never raise this bar.
    best_receipted = max((r.get("local_heldout_valid_accuracy") or 0 for r in receipts), default=0)
    min_valid = args.min_valid if args.min_valid is not None else best_receipted - 0.0015

    listed = kaggle_submissions(Path(args.kaggle))
    by_ref = {r["ref"]: r for r in listed}
    # Fill in scores that were still pending when their receipt was written.
    refreshed = 0
    for receipt in receipts:
        live = by_ref.get(receipt["submission_ref"])
        if live and receipt.get("public_score") is None and live["public_score"] is not None:
            receipt["public_score"], receipt["status"] = live["public_score"], live["status"]
            refreshed += 1
    if refreshed and not args.dry_run:
        receipts_path.write_text("".join(json.dumps(r) + "\n" for r in receipts))
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    used_today = sum(1 for r in listed if r["date"].startswith(today))
    left = max(0, DAILY_QUOTA - used_today)
    print(f"{used_today} of {DAILY_QUOTA} submissions used on {today} UTC; {left} left (reserve {args.reserve}); "
          f"{refreshed} receipts refreshed")

    # Best (own or twin) held-out accuracy first, so a scarce quota goes to the strongest new candidate; ties by
    # delivery time.
    checked = [(d,) + check_candidate(d, sample_ids, twins) for d in Path(args.candidates).iterdir()
               if (d / "submission.csv").exists() and (d / "submission.audit.json").exists()]
    checked.sort(key=lambda item: (-item[4]["proxy"], (item[0] / "submission.csv").stat().st_mtime))
    uploaded = 0
    for directory, digest, audit, problems, info in checked:
        if digest in known:
            continue
        if problems:
            print(f"SKIP {directory.name}: " + "; ".join(problems))
            continue
        valid, full, twin = info["proxy"], info["full"], info["twin"]
        floor = min_valid - args.full_margin if full else min_valid
        if valid < floor and directory.name not in args.force:
            basis = f"twin {twin} held-out" if full else "held-out"
            print(f"SKIP {directory.name}: {basis} {valid:.4f} below {floor:.4f} (pass --force {directory.name} to upload anyway)")
            continue
        # A new best is what the reserve is kept for, so it is never held back, and a full-training candidate is
        # a deliverable rather than an experiment. A candidate merely close to the best leaves the reserve, and
        # one clearly weaker also leaves room for a stronger candidate delivered later the same UTC day.
        if full or valid > best_receipted or directory.name in args.force:
            needed = 1
        elif valid >= best_receipted - args.margin:
            needed = 1 + args.reserve
        else:
            needed = 1 + args.reserve + args.keep_for_better
        if uploaded >= args.max_uploads or left < needed:
            print(f"WAIT {directory.name}: valid, but {left} submission(s) left today and it needs {needed}")
            continue
        message, prediction = describe(directory, audit, info)
        if args.dry_run:
            print(f"WOULD UPLOAD {directory.name}: {message}")
            uploaded += 1
            left -= 1
            if not full:
                best_receipted = max(best_receipted, valid)
            continue
        before = {r["ref"] for r in listed}
        result = subprocess.run([str(args.kaggle), "competitions", "submit", "-c", COMPETITION,
                                 "-f", str(directory / "submission.csv"), "-m", message],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or "Successfully submitted" not in result.stdout:
            print(f"FAILED {directory.name}: {result.stderr.strip() or result.stdout.strip()}")
            break
        submitted_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        live = None
        for _ in range(8):
            time.sleep(10)
            listed = kaggle_submissions(Path(args.kaggle))
            new = [r for r in listed if r["ref"] not in before]
            if new and new[0]["public_score"] is not None:
                live = new[0]
                break
            live = new[0] if new else live
        config = candidate_config(directory)
        receipt = {"submission_ref": live["ref"] if live else None, "submitted_utc": submitted_utc,
                   "file": str(directory / "submission.csv"), "csv_sha256": digest,
                   "checkpoint": audit.get("checkpoint", f"artifacts/candidates/{directory.name}/last.pt"),
                   "arch": audit.get("arch"), "parameter_count": audit.get("parameter_count"),
                   "weights": audit.get("weights", "ema"), "tta": audit.get("tta", "flip"),
                   "full_train": full, "train_images": audit.get("train_images", config.get("train_images")),
                   "local_heldout_valid_accuracy": None if full else audit.get("valid_accuracy_of_these_weights"),
                   "status": live["status"] if live else "PENDING",
                   "public_score": live["public_score"] if live else None, "private_score": None,
                   "purpose": (f"HPC full-training candidate {directory.name} (final-submission deliverable; "
                               f"generalization estimate from twin {twin})" if full else
                               f"HPC candidate {directory.name} (automatic upload of a delivered candidate)"),
                   "preregistered_public_prediction_pct": prediction, "preregistered_band_pp": 2.0,
                   "selected_for_final": "no"}
        if full:
            receipt.update(twin=twin, twin_heldout_valid_accuracy=valid,
                           fit_accuracy_on_trained_split=audit.get("valid_accuracy_of_these_weights"))
        with receipts_path.open("a") as handle:
            handle.write(json.dumps(receipt) + "\n")
        known.add(digest)
        if not full:
            best_receipted = max(best_receipted, valid)
        uploaded += 1
        left -= 1
        basis = f"twin {twin} held-out {valid}" if full else f"val {valid}"
        print(f"UPLOADED {directory.name}: ref {receipt['submission_ref']} public {receipt['public_score']} "
              f"({basis}, predicted {prediction})")
    if uploaded == 0:
        print("nothing new to upload")


if __name__ == "__main__":
    main()
