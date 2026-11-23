from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path


def sha256_digest(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact missing for SHA-256 audit: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_verified_table(
    manifest_path: str | Path,
    raw_csv_path: str | Path,
    latex_table_path: str | Path,
    measured_rows: int,
    experiment_id: str,
) -> dict:
    if int(measured_rows) <= 0:
        raise ValueError("Zero-Estimation Violation: measured_rows must be > 0 from real execution.")
    raw_hash = sha256_digest(raw_csv_path)
    tex_hash = sha256_digest(latex_table_path)
    record = {
        "experiment_id": str(experiment_id),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "raw_csv_path": str(raw_csv_path),
        "raw_csv_sha256": raw_hash,
        "measured_rows": int(measured_rows),
        "latex_table_path": str(latex_table_path),
        "latex_table_sha256": tex_hash,
    }
    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    if mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        payload = {"verified_artifacts": []}
    payload["verified_artifacts"] = [
        x for x in payload.get("verified_artifacts", []) if x.get("experiment_id") != experiment_id
    ]
    payload["verified_artifacts"].append(record)
    mpath.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return record