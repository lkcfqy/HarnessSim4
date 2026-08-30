"""Integrity audit for the public FAU/FAPS Stripped Wire Dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from harnessbench.inspect_public_data import file_sha256

STRIPPED_WIRE_RECORD_URL = "https://zenodo.org/records/16686806"
STRIPPED_WIRE_API_URL = "https://zenodo.org/api/records/16686806"
STRIPPED_WIRE_DOI = "10.5281/zenodo.16686806"
STRIPPED_WIRE_VERSION = "v2"
STRIPPED_WIRE_LICENSE = "CC BY 4.0"
STRIPPED_WIRE_ZIP_MD5 = "9e1944f1b7c9bc9e7db433f7bdcf2694"
STRIPPED_WIRE_ZIP_SHA256 = "22754426bae0d15420bd7a10198dadf3dc2925a81e5b2fdb6dad95b5d05d3cbd"
EXPECTED_COUNTS = {
    "train/PatchCore": 200,
    "train/VLM": 3,
    "test/good": 133,
    "test/cut_strands": 68,
    "test/pulled_strands": 99,
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_stripped_wire(
    dataset_root: Path,
    output_dir: Path,
    *,
    zip_path: Path | None = None,
) -> dict:
    """Validate counts, image metadata, hashes, and train/test disjointness."""

    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(dataset_root.rglob("*.jpg"))
    counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    rows = []
    for path in image_paths:
        relative = path.relative_to(dataset_root).as_posix()
        split_class = path.parent.relative_to(dataset_root).as_posix()
        counts[split_class] += 1
        sha256 = file_sha256(path)
        hashes[sha256].append(relative)
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        shapes[f"{width}x{height}"] += 1
        modes[mode] += 1
        rows.append(
            {
                "relative_path": relative,
                "split_class": split_class,
                "width": width,
                "height": height,
                "mode": mode,
                "bytes": path.stat().st_size,
                "sha256": sha256,
            }
        )

    observed_counts = dict(sorted(counts.items()))
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected Stripped Wire counts: {observed_counts}")
    if len(image_paths) != 503:
        raise ValueError(f"expected 503 JPEG files, found {len(image_paths)}")
    if set(modes) != {"RGB"}:
        raise ValueError(f"expected RGB images only, found {dict(modes)}")
    if any(int(shape.split("x")[1]) != 1079 for shape in shapes):
        raise ValueError("all released images are expected to have height 1079")

    train_hashes = {
        row["sha256"] for row in rows if row["split_class"] == "train/PatchCore"
    }
    test_hashes = {row["sha256"] for row in rows if row["split_class"].startswith("test/")}
    train_test_overlap = sorted(train_hashes & test_hashes)
    if train_test_overlap:
        raise ValueError("exact image duplicate found between PatchCore train and test")

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    manifest = "\n".join(f'{row["relative_path"]} {row["sha256"]}' for row in rows)
    tree_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    index_path = output_dir / "stripped_wire_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    archive = None
    if zip_path is not None:
        zip_path = zip_path.resolve()
        archive = {
            "path": str(zip_path),
            "bytes": zip_path.stat().st_size,
            "md5": _md5(zip_path),
            "sha256": file_sha256(zip_path),
            "expected_md5": STRIPPED_WIRE_ZIP_MD5,
            "expected_sha256": STRIPPED_WIRE_ZIP_SHA256,
        }
        if archive["md5"] != STRIPPED_WIRE_ZIP_MD5:
            raise ValueError("downloaded Stripped Wire archive MD5 mismatch")
        if archive["sha256"] != STRIPPED_WIRE_ZIP_SHA256:
            raise ValueError("downloaded Stripped Wire archive SHA-256 mismatch")

    report = {
        "schema_version": 1,
        "name": "FAU/FAPS Stripped Wire Dataset integrity audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "source_record": STRIPPED_WIRE_RECORD_URL,
        "source_api": STRIPPED_WIRE_API_URL,
        "doi": STRIPPED_WIRE_DOI,
        "release_version": STRIPPED_WIRE_VERSION,
        "license": STRIPPED_WIRE_LICENSE,
        "archive": archive,
        "counts": observed_counts,
        "total_images": len(image_paths),
        "image_modes": dict(sorted(modes.items())),
        "unique_shapes": len(shapes),
        "width_range": [min(row["width"] for row in rows), max(row["width"] for row in rows)],
        "height_values": sorted({row["height"] for row in rows}),
        "tree_sha256": tree_sha256,
        "train_test_exact_duplicate_count": len(train_test_overlap),
        "all_duplicate_groups": duplicate_groups,
        "duplicate_note": (
            "The sole exact duplicate is a PatchCore training image copied into the optional "
            "three-image VLM reference folder; the VLM folder is excluded from all experiments."
        ),
        "index_csv": str(index_path.resolve()),
        "index_sha256": file_sha256(index_path),
    }
    report_path = output_dir / "stripped_wire_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
