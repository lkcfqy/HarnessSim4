from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from harnessbench.learning.inspect_perception import InspectPerceptionConfig, preprocess_image
from harnessbench.learning.inspect_wire_external import _spatial_nn_scores


def test_aspect_preserving_preprocessing_letterboxes_narrow_wire(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 100), (220, 220, 220))
    for x in range(8, 12):
        for y in range(100):
            image.putpixel((x, y), (220, 20, 20))
    path = tmp_path / "wire.jpg"
    image.save(path, quality=100)
    config = InspectPerceptionConfig(
        crop=100,
        feature_size=13,
        preserve_aspect_ratio=True,
    )
    tensor = preprocess_image(path, config)
    assert tensor.shape == (3, 100, 100)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = tensor * std + mean
    red_dominant = (rgb[0] > rgb[1] + 0.2).float().mean().item()
    assert 0.01 < red_dominant < 0.15


def test_spatial_nn_is_zero_for_reference_copy() -> None:
    config = InspectPerceptionConfig(
        feature_size=2,
        selected_channels=2,
        image_score_top_fraction=0.25,
    )
    reference = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]],
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
        ]
    )
    scores, maps = _spatial_nn_scores(
        reference,
        reference[:1],
        config,
        device_name="cpu",
        batch_size=1,
    )
    np.testing.assert_allclose(scores, 0.0)
    np.testing.assert_allclose(maps, 0.0)


def test_spatial_nn_rejects_incompatible_feature_size() -> None:
    config = InspectPerceptionConfig(feature_size=3, selected_channels=2)
    reference = torch.zeros((2, 2, 2, 2))
    with pytest.raises(RuntimeError):
        _spatial_nn_scores(reference, reference[:1], config, device_name="cpu")
