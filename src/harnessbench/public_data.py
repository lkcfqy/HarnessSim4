"""Download the pinned public proxy dataset without requiring a Hugging Face client."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

DATASET_ID = "lerobot/aloha_sim_insertion_human"
REVISION = "main"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOTAL_ROWS = 25_000
ROWS_PER_PAGE = 100
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    required: bool = True


FILES = (
    RemoteFile("README.md", 4_138),
    RemoteFile("meta/info.json", 2_676),
    RemoteFile("meta/stats.json", 6_987),
    RemoteFile("meta/tasks.parquet", 2_185),
    RemoteFile("meta/episodes/chunk-000/file-000.parquet", 88_256),
    RemoteFile("data/chunk-000/file-000.parquet", 855_778),
    RemoteFile("data/chunk-000/file-001.parquet", 847_617),
    RemoteFile("data/chunk-000/file-002.parquet", 855_579),
    RemoteFile("data/chunk-000/file-003.parquet", 293_268),
    RemoteFile("videos/observation.images.top/chunk-000/file-000.mp4", 88_389_634, required=False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "HarnessBench-Insert/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _read_json_url(url: str, attempts: int = 5) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "HarnessBench-Insert/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - transport failures are retryable
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_error


def _page_url(offset: int, length: int) -> str:
    query = urlencode(
        {
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_API}?{query}"


def _fetch_api_page(page_dir: Path, offset: int) -> tuple[int, Path]:
    length = min(ROWS_PER_PAGE, TOTAL_ROWS - offset)
    target = page_dir / f"rows_{offset:06d}.json"
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if len(cached.get("rows", [])) == length:
                return offset, target
        except (OSError, json.JSONDecodeError):
            pass

    payload = _read_json_url(_page_url(offset, length))
    if payload.get("num_rows_total") != TOTAL_ROWS or len(payload.get("rows", [])) != length:
        raise RuntimeError(f"Unexpected Dataset Viewer response at offset {offset}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".json.part")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
    partial.replace(target)
    return offset, target


def _cache_record_if_valid(output_dir: Path) -> dict | None:
    cache_path = output_dir / "data" / "harnessbench_rows.npz"
    metadata_path = output_dir / "data" / "harnessbench_rows.meta.json"
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as arrays:
            if (
                arrays["state"].shape == (TOTAL_ROWS, 14)
                and arrays["action"].shape == (TOTAL_ROWS, 14)
                and arrays["episode"].shape == (TOTAL_ROWS,)
                and arrays["frame"].shape == (TOTAL_ROWS,)
                and arrays["timestamp"].shape == (TOTAL_ROWS,)
            ):
                source = "portable cache created from the pinned public dataset"
                if metadata_path.exists():
                    try:
                        with metadata_path.open("r", encoding="utf-8") as handle:
                            source = json.load(handle).get("source", source)
                    except (OSError, json.JSONDecodeError):
                        pass
                return {
                    "status": "cached",
                    "path": str(cache_path.relative_to(output_dir).as_posix()),
                    "rows": TOTAL_ROWS,
                    "sha256": _sha256(cache_path),
                    "source": source,
                }
    except (OSError, KeyError, ValueError):
        return None
    return None


def _write_numpy_cache(
    output_dir: Path,
    state: np.ndarray,
    action: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    timestamp: np.ndarray,
    source: str,
) -> dict:
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    episode = np.asarray(episode, dtype=np.int64)
    frame = np.asarray(frame, dtype=np.int64)
    timestamp = np.asarray(timestamp, dtype=np.float32)
    if state.shape != (TOTAL_ROWS, 14) or action.shape != (TOTAL_ROWS, 14):
        raise RuntimeError(f"Unexpected state/action shapes: {state.shape} and {action.shape}")
    if any(array.shape != (TOTAL_ROWS,) for array in (episode, frame, timestamp)):
        raise RuntimeError("Unexpected episode/frame/timestamp shapes")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError("Public data contains non-finite state/action values")

    order = np.lexsort((frame, episode))
    cache_path = output_dir / "data" / "harnessbench_rows.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(".npz.part")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            state=state[order],
            action=action[order],
            episode=episode[order],
            frame=frame[order],
            timestamp=timestamp[order],
        )
    partial.replace(cache_path)
    record = {
        "status": "created",
        "path": str(cache_path.relative_to(output_dir).as_posix()),
        "rows": TOTAL_ROWS,
        "sha256": _sha256(cache_path),
        "source": source,
    }
    metadata_path = output_dir / "data" / "harnessbench_rows.meta.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": source,
                "rows": TOTAL_ROWS,
                "sha256": record["sha256"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    return record


def _node_executable() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("HARNESSBENCH_NODE")
    if configured:
        candidates.append(Path(configured))
    on_path = shutil.which("node")
    if on_path:
        candidates.append(Path(on_path))
    # Codex Desktop's bundled layout: dependencies/python/python.exe -> dependencies/node/bin/node.exe
    candidates.append(Path(sys.executable).resolve().parent.parent / "node" / "bin" / "node.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_numpy_cache_from_node(output_dir: Path) -> dict | None:
    """Use the pure-JavaScript reader when native Parquet DLLs are unavailable."""

    existing = _cache_record_if_valid(output_dir)
    if existing:
        return existing
    node = _node_executable()
    script = PROJECT_ROOT / "tools" / "parquet_to_jsonl.mjs"
    module = PROJECT_ROOT / "node_modules" / "hyparquet" / "package.json"
    if node is None or not script.exists() or not module.exists():
        return None

    jsonl_path = output_dir / "data" / "harnessbench_rows.jsonl.part"
    command = [str(node), str(script), str(output_dir), str(jsonl_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        states: list[list[float]] = []
        actions: list[list[float]] = []
        episodes: list[int] = []
        frames: list[int] = []
        timestamps: list[float] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                states.append(row["observation.state"])
                actions.append(row["action"])
                episodes.append(int(row["episode_index"]))
                frames.append(int(row["frame_index"]))
                timestamps.append(float(row["timestamp"]))
        return _write_numpy_cache(
            output_dir,
            np.asarray(states),
            np.asarray(actions),
            np.asarray(episodes),
            np.asarray(frames),
            np.asarray(timestamps),
            source="local pinned Parquet decoded with pure-JavaScript hyparquet",
        )
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return None
    finally:
        jsonl_path.unlink(missing_ok=True)


def build_numpy_cache_from_pyarrow(output_dir: Path) -> dict | None:
    """Use PyArrow when its native library is permitted by the host."""

    existing = _cache_record_if_valid(output_dir)
    if existing:
        return existing
    try:
        import pyarrow.parquet as pq

        columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
        chunks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
        files = sorted((output_dir / "data").glob("chunk-*/*.parquet"))
        for path in files:
            values = pq.read_table(path, columns=columns).to_pydict()
            for column in columns:
                chunks[column].append(np.asarray(values[column]))
        if not files:
            return None
        return _write_numpy_cache(
            output_dir,
            np.concatenate(chunks["observation.state"]),
            np.concatenate(chunks["action"]),
            np.concatenate(chunks["episode_index"]),
            np.concatenate(chunks["frame_index"]),
            np.concatenate(chunks["timestamp"]),
            source="local pinned Parquet decoded with PyArrow",
        )
    except Exception:  # noqa: BLE001 - optional local decoder probe
        return None


def build_numpy_cache_from_api(output_dir: Path, workers: int = 12) -> dict:
    """Build a portable state-action cache through the official Dataset Viewer API."""

    existing = _cache_record_if_valid(output_dir)
    if existing:
        return existing

    page_dir = output_dir / "api_pages"
    offsets = list(range(0, TOTAL_ROWS, ROWS_PER_PAGE))
    pages: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch_api_page, page_dir, offset): offset for offset in offsets}
        for future in as_completed(futures):
            offset, path = future.result()
            pages[offset] = path

    states: list[list[float]] = []
    actions: list[list[float]] = []
    episodes: list[int] = []
    frames: list[int] = []
    timestamps: list[float] = []
    expected_row = 0
    for offset in offsets:
        with pages[offset].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for wrapped in payload["rows"]:
            if int(wrapped["row_idx"]) != expected_row:
                raise RuntimeError(
                    f"Dataset Viewer row discontinuity: expected {expected_row}, "
                    f"got {wrapped['row_idx']}"
                )
            row = wrapped["row"]
            states.append(row["observation.state"])
            actions.append(row["action"])
            episodes.append(int(row["episode_index"]))
            frames.append(int(row["frame_index"]))
            timestamps.append(float(row["timestamp"]))
            expected_row += 1

    return _write_numpy_cache(
        output_dir,
        np.asarray(states),
        np.asarray(actions),
        np.asarray(episodes),
        np.asarray(frames),
        np.asarray(timestamps),
        source=ROWS_API,
    )


def build_portable_numpy_cache(output_dir: Path) -> dict:
    existing = _cache_record_if_valid(output_dir)
    if existing:
        return existing
    for builder in (build_numpy_cache_from_node, build_numpy_cache_from_pyarrow):
        result = builder(output_dir)
        if result is not None:
            return result
    return build_numpy_cache_from_api(output_dir, workers=4)


def download_public_dataset(output_dir: Path, include_video: bool = False) -> dict:
    """Download and verify the files required by the state-action baseline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for remote in FILES:
        if not remote.required and not include_video:
            continue
        target = output_dir / remote.path
        status = "cached"
        if not target.exists() or target.stat().st_size != remote.size:
            _download(BASE_URL + remote.path, target)
            status = "downloaded"
        actual_size = target.stat().st_size
        if actual_size != remote.size:
            raise RuntimeError(
                f"Size mismatch for {remote.path}: expected {remote.size}, got {actual_size}. "
                "The upstream dataset may have changed; inspect before updating the pin."
            )
        records.append(
            {
                **asdict(remote),
                "status": status,
                "sha256": _sha256(target),
            }
        )

    portable_cache = build_portable_numpy_cache(output_dir)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "revision": REVISION,
        "license": "MIT",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_video": include_video,
        "files": records,
        "portable_numpy_cache": portable_cache,
    }
    with (output_dir / "harnessbench_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest
