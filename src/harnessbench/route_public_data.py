"""Pinned Berkeley real-robot cable-routing dataset adapter.

Only the compact LeRobot state/action parquet and metadata are required.  The
four 128x128 camera streams are optional and remain byte-for-byte upstream
assets.  Every downloaded file is pinned to an immutable repository revision
and recorded in a local SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harnessbench.lerobot import TrajectoryTable, dataset_summary, load_trajectories

DATASET_ID = "lerobot/berkeley_cable_routing"
REVISION = "20a7774a714a5daf56b220e01635425f2cb9745b"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}"
ORIGINAL_PROJECT = "https://sites.google.com/view/cablerouting/data"
EXPECTED_EPISODES = 1_647
EXPECTED_FRAMES = 42_328


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None


CORE_FILES = (
    RemoteFile("README.md", 5_631),
    RemoteFile("meta/info.json", 4_747),
    RemoteFile("meta/stats.json", 9_420),
    RemoteFile(
        "meta/tasks.parquet",
        2_045,
        "86f8fccd7efff90d518b23ac254d29b3683319f7b917763abc9e4d62e7f79f8f",
    ),
    RemoteFile(
        "meta/episodes/chunk-000/file-000.parquet",
        2_167_447,
        "56827e74c4f803ae6aafe54b1b10167229e6c9dd1501e547a787f2a244773915",
    ),
    RemoteFile(
        "data/chunk-000/file-000.parquet",
        3_241_893,
        "424ba5a986a4f85d0f286e668b647042bb8e5ba32f2bba4940c494c299be448d",
    ),
)

VIDEO_FILES = (
    RemoteFile(
        "videos/observation.images.image/chunk-000/file-000.mp4",
        63_576_401,
        "35da56b96b70cf96d913d7a4fedb0d97555778639eccdd5486879e54bfb7713c",
    ),
    RemoteFile(
        "videos/observation.images.top_image/chunk-000/file-000.mp4",
        98_289_042,
        "e4b2933355a595369b4085205c60cbef4690969c345e5287730ff9a38e3a97f1",
    ),
    RemoteFile(
        "videos/observation.images.wrist225_image/chunk-000/file-000.mp4",
        73_485_154,
        "a28ef8386fc049ca1b8122cb43fb90e8833b389b02e7eb22900cf44038c4d359",
    ),
    RemoteFile(
        "videos/observation.images.wrist45_image/chunk-000/file-000.mp4",
        80_087_238,
        "0ce4b60183b900a6a66751daaf5837c5d792cd4927d2b52b2bb693ce2ea9670e",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path, status: str) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "status": status,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "url": BASE_URL + path.relative_to(root).as_posix(),
    }


def _cache_record(path: Path, root: Path, status: str) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "status": status,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "source": "generated locally from the pinned parquet state/action columns",
    }


def _download_file(remote: RemoteFile, output_dir: Path) -> dict:
    target = output_dir / remote.path
    if target.exists():
        actual_size = target.stat().st_size
        actual_hash = _sha256(target) if remote.sha256 else None
        if actual_size != remote.size or (
            remote.sha256 is not None and actual_hash != remote.sha256
        ):
            raise RuntimeError(
                f"Existing public-data file failed integrity validation: {target}. "
                "Move it aside explicitly before retrying."
            )
        return _file_record(target, output_dir, "cached")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        BASE_URL + remote.path,
        headers={"User-Agent": "HarnessSim4-RouteBot/0.3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if partial.stat().st_size != remote.size:
            raise RuntimeError(
                f"Unexpected byte count for {remote.path}: "
                f"{partial.stat().st_size} != {remote.size}"
            )
        if remote.sha256 is not None and _sha256(partial) != remote.sha256:
            raise RuntimeError(f"SHA-256 mismatch for {remote.path}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return _file_record(target, output_dir, "downloaded")


def validate_route_table(table: TrajectoryTable, *, strict_size: bool = True) -> dict:
    summary = dataset_summary(table)
    if table.state.shape[1] != 8 or table.action.shape[1] != 7:
        raise ValueError(
            f"Unexpected Berkeley state/action dimensions: {table.state.shape[1]} and "
            f"{table.action.shape[1]}"
        )
    if not summary["finite"]:
        raise ValueError("Berkeley cable-routing data contains non-finite state/action values")
    if strict_size and (
        summary["episodes"] != EXPECTED_EPISODES or summary["frames"] != EXPECTED_FRAMES
    ):
        raise ValueError(
            "Pinned Berkeley dataset cardinality changed: "
            f"{summary['episodes']} episodes and {summary['frames']} frames"
        )
    pairs = np.column_stack((table.episode, table.frame))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate (episode, frame) keys detected")
    expected_ids = np.arange(table.episodes.min(), table.episodes.max() + 1)
    if not np.array_equal(table.episodes, expected_ids):
        raise ValueError("Episode identifiers are not contiguous")
    return {
        **summary,
        "episode_ids_contiguous": True,
        "episode_frame_keys_unique": True,
        "source_domain": "real robot teleoperation",
        "camera_views_available": 4,
    }


def _write_numpy_cache(output_dir: Path, table: TrajectoryTable) -> dict:
    cache_path = output_dir / "data" / "harnessbench_rows.npz"
    if cache_path.exists():
        with np.load(cache_path) as arrays:
            cached_rows = int(arrays["state"].shape[0])
        if cached_rows != EXPECTED_FRAMES:
            raise RuntimeError(
                f"Existing route cache has {cached_rows} rows; expected {EXPECTED_FRAMES}"
            )
        return _cache_record(cache_path, output_dir, "cached")

    partial = cache_path.with_suffix(".npz.part")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            state=np.asarray(table.state, dtype=np.float32),
            action=np.asarray(table.action, dtype=np.float32),
            episode=np.asarray(table.episode, dtype=np.int64),
            frame=np.asarray(table.frame, dtype=np.int64),
            timestamp=np.asarray(table.timestamp, dtype=np.float32),
        )
    partial.replace(cache_path)
    return _cache_record(cache_path, output_dir, "created")


def _write_data_card(output_dir: Path, summary: dict, include_videos: bool) -> None:
    card = f"""# Berkeley Cable Routing data card for RouteBot

- Upstream project: {ORIGINAL_PROJECT}
- Pinned LeRobot conversion: {DATASET_PAGE}
- Immutable revision: `{REVISION}`
- Domain: real Franka Panda teleoperation for single- and multi-clip cable routing
- Local rows: {summary['frames']:,} frames in {summary['episodes']:,} whole episodes
- State/action dimensions: {summary['state_dim']} / {summary['action_dim']}
- Camera streams downloaded: {'yes (four views)' if include_videos else 'no'}

## License and attribution

The original project page releases the data under CC BY 4.0. The LeRobot
conversion card labels the converted repository Apache-2.0. RouteBot therefore
records both notices and uses the more conservative attribution requirement.

## Claim boundary

This public dataset contains real robot observations and actions, but it does
not expose RouteBot's segment-to-clip relation intervention or a standardized
closed-loop semantic-success label. It can validate data ingestion and offline
action prediction; it cannot validate RouteBot's causal semantic-routing claim
or establish production success.
"""
    path = output_dir / "DATACARD.md"
    path.write_text(card, encoding="utf-8")


def download_route_public_dataset(
    output_dir: Path,
    *,
    include_videos: bool = False,
    workers: int = 4,
) -> dict:
    """Download, hash, cache, and validate the pinned real cable-routing data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    remote_files = (*CORE_FILES, *(VIDEO_FILES if include_videos else ()))
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(remote_files)))) as executor:
        futures = {
            executor.submit(_download_file, remote, output_dir): remote for remote in remote_files
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["path"])

    table = load_trajectories(output_dir)
    summary = validate_route_table(table)
    cache_record = _write_numpy_cache(output_dir, table)
    _write_data_card(output_dir, summary, include_videos)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_project": ORIGINAL_PROJECT,
        "upstream_project_license": "CC-BY-4.0",
        "conversion_repository_license": "Apache-2.0",
        "source_domain": "real robot teleoperation",
        "files": records,
        "portable_numpy_cache": cache_record,
        "dataset": summary,
        "videos_included": include_videos,
        "remote_inventory": [asdict(item) for item in remote_files],
    }
    manifest_path = output_dir / "harnessbench_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def inspect_route_public_dataset(output_dir: Path) -> dict:
    table = load_trajectories(output_dir)
    return {
        "data_dir": str(output_dir.resolve()),
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        **validate_route_table(table),
    }
