"""Audit and index the public MVTec AD ``cable`` category.

The original MVTec AD data are distributed under CC BY-NC-SA 4.0.  The
project downloads the category from a pinned Hugging Face transport mirror
because the current MVTec download page requires a personal-information form.
The mirror is never described as an additional independent dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

MVTEC_AD_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
MVTEC_AD_PAPER = "https://doi.org/10.1109/CVPR.2019.00982"
MVTEC_CABLE_MIRROR = "https://huggingface.co/datasets/foersben/mvtec-ad"
MVTEC_CABLE_MIRROR_REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
MVTEC_LICENSE = "CC BY-NC-SA 4.0"

EXPECTED_TRAIN_GOOD = 224
EXPECTED_TEST_GOOD = 58
EXPECTED_DEFECT_COUNTS = {
    "bent_wire": 13,
    "cable_swap": 12,
    "combined": 11,
    "cut_inner_insulation": 14,
    "cut_outer_insulation": 10,
    "missing_cable": 12,
    "missing_wire": 10,
    "poke_insulation": 10,
}


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["relative_path"])):
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _image_record(root: Path, image_path: Path) -> dict:
    relative = image_path.relative_to(root).as_posix()
    parts = image_path.relative_to(root).parts
    split = parts[0]
    if split == "train":
        defect_type = "good"
    elif split == "test":
        defect_type = parts[1]
    else:
        raise ValueError(f"unexpected MVTec image path: {relative}")
    with Image.open(image_path) as image:
        width, height = image.size
        mode = image.mode
    mask_path = None
    if split == "test" and defect_type != "good":
        candidate = root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        if not candidate.is_file():
            raise FileNotFoundError(f"missing mask for {relative}: {candidate}")
        mask_path = candidate.relative_to(root).as_posix()
        with Image.open(candidate) as mask:
            if mask.size != (width, height):
                raise ValueError(
                    f"mask/image size mismatch for {relative}: {mask.size} vs {(width, height)}"
                )
    return {
        "relative_path": relative,
        "split": split,
        "defect_type": defect_type,
        "is_anomaly": defect_type != "good",
        "mask_relative_path": mask_path,
        "width": width,
        "height": height,
        "mode": mode,
        "bytes": image_path.stat().st_size,
        "sha256": file_sha256(image_path),
    }


def audit_mvtec_cable(root: Path, output_dir: Path | None = None) -> dict:
    """Validate the cable category and optionally write an index and report."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    license_path = root / "license.txt"
    readme_path = root / "readme.txt"
    if not license_path.is_file() or not readme_path.is_file():
        raise FileNotFoundError("MVTec cable license.txt/readme.txt are required")
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in license_text:
        raise ValueError("unexpected MVTec cable license text")

    image_paths = sorted((root / "train" / "good").glob("*.png"))
    image_paths += sorted((root / "test").glob("*/*.png"))
    records = [_image_record(root, path) for path in image_paths]
    train_good = sum(row["split"] == "train" for row in records)
    test_good = sum(
        row["split"] == "test" and row["defect_type"] == "good" for row in records
    )
    defect_counts = Counter(
        str(row["defect_type"])
        for row in records
        if row["split"] == "test" and row["is_anomaly"]
    )
    if train_good != EXPECTED_TRAIN_GOOD:
        raise ValueError(f"expected {EXPECTED_TRAIN_GOOD} train/good images, found {train_good}")
    if test_good != EXPECTED_TEST_GOOD:
        raise ValueError(f"expected {EXPECTED_TEST_GOOD} test/good images, found {test_good}")
    if dict(sorted(defect_counts.items())) != EXPECTED_DEFECT_COUNTS:
        raise ValueError(f"unexpected defect counts: {dict(sorted(defect_counts.items()))}")

    mask_paths = sorted((root / "ground_truth").glob("*/*_mask.png"))
    expected_masks = sum(EXPECTED_DEFECT_COUNTS.values())
    if len(mask_paths) != expected_masks:
        raise ValueError(f"expected {expected_masks} masks, found {len(mask_paths)}")
    auxiliary_rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in [*mask_paths, license_path, readme_path]
    ]
    tree_rows = [*records, *auxiliary_rows]
    report = {
        "schema_version": 1,
        "name": "MVTec AD cable public-data audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "source_dataset": "MVTec AD",
        "source_category": "cable",
        "source_page": MVTEC_AD_PAGE,
        "source_paper": MVTEC_AD_PAPER,
        "license": MVTEC_LICENSE,
        "transport_mirror": MVTEC_CABLE_MIRROR,
        "transport_mirror_revision": MVTEC_CABLE_MIRROR_REVISION,
        "claim_scope": (
            "Real cable cross-section anomaly perception only; the data do not contain "
            "automotive harness topology, robot viewpoints, or active-inspection trajectories."
        ),
        "counts": {
            "train_good": train_good,
            "test_good": test_good,
            "test_anomaly": sum(defect_counts.values()),
            "masks": len(mask_paths),
            "indexed_images": len(records),
            "all_files": len(tree_rows),
        },
        "defect_counts": dict(sorted(defect_counts.items())),
        "image_shapes": sorted(
            {
                f"{int(row['width'])}x{int(row['height'])}:{row['mode']}"
                for row in records
            }
        ),
        "license_sha256": file_sha256(license_path),
        "readme_sha256": file_sha256(readme_path),
        "tree_sha256": _tree_sha256(tree_rows),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / "mvtec_cable_index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        report["index_csv"] = str(index_path.resolve())
        report_path = output_dir / "mvtec_cable_audit.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["report"] = str(report_path.resolve())
    return report

