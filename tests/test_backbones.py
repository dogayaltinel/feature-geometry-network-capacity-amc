from __future__ import annotations

import pytest
import torch

from backbones import WIDTHS, build_model, count_parameters, scale_channels


@pytest.mark.parametrize("backbone", ["BB1", "BB2"])
@pytest.mark.parametrize("feature", ["F2", "F3", "F5", "F7"])
@pytest.mark.parametrize("width", WIDTHS)
def test_all_declared_cells_emit_eleven_logits(backbone: str, feature: str, width: float) -> None:
    model, spec = build_model(backbone, feature, width)
    x = torch.randn(2, spec.input_channels, 128)
    output = model(x)
    assert output.shape == (2, 11)
    trainable, total = count_parameters(model)
    assert trainable == total and total > 0


def test_channel_rounding_is_shared() -> None:
    assert scale_channels((24, 32, 48), 0.5) == (12, 16, 24)
    assert all(channel % 4 == 0 for channel in scale_channels((24, 32, 48), 1.5))
