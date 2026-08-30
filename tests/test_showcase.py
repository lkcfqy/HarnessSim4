from pathlib import Path

from PIL import Image

from scripts.render_harnesssim4_showcase import (
    TASKS,
    _grid_frame,
    _sample_index,
    _timestamp,
    _write_vtt,
)


def test_sample_index_maps_both_endpoints_monotonically() -> None:
    values = [_sample_index(index, 11, 4) for index in range(11)]
    assert values[0] == 0
    assert values[-1] == 3
    assert values == sorted(values)


def test_grid_frame_has_requested_size_and_all_quadrants() -> None:
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    task_frames = {
        spec.key: Image.new("RGB", (80, 45), color) for spec, color in zip(TASKS, colors)
    }
    frame = _grid_frame(task_frames, (320, 180))
    assert frame.size == (320, 180)
    assert frame.getpixel((40, 30)) == colors[0]
    assert frame.getpixel((240, 30)) == colors[1]
    assert frame.getpixel((40, 140)) == colors[2]
    assert frame.getpixel((240, 140)) == colors[3]


def test_vtt_uses_nonoverlapping_chapter_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "chapters.vtt"
    chapters = [
        {"label": "Overview", "start_s": 0.0},
        {"label": "Inspect", "start_s": 2.5},
    ]
    _write_vtt(path, chapters, 8.0)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert f"{_timestamp(0.0)} --> {_timestamp(2.5)}" in text
    assert f"{_timestamp(2.5)} --> {_timestamp(8.0)}" in text
