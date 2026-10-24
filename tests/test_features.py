from __future__ import annotations

import torch

from features import FEATURE_CHANNELS, derive_features, rms_normalize_iq


def test_feature_shapes_and_finite_values() -> None:
    x = torch.randn(4, 2, 128)
    x = rms_normalize_iq(x)
    for code, channels in FEATURE_CHANNELS.items():
        output = derive_features(x, code)
        assert output.shape == (4, channels, 128)
        assert torch.isfinite(output).all()


def test_rms_normalization() -> None:
    x = torch.randn(3, 2, 128) * 7.0
    normalized = rms_normalize_iq(x)
    rms = torch.sqrt(normalized.square().mean(dim=(1, 2)))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1.0e-5)


def test_f7_difference_first_sample_is_zero() -> None:
    x = rms_normalize_iq(torch.randn(2, 2, 128))
    output = derive_features(x, "F7")
    assert torch.allclose(output[:, 5:, 0], torch.zeros_like(output[:, 5:, 0]))


def test_known_constant_signal() -> None:
    x = torch.zeros(1, 2, 8)
    x[:, 0, :] = 1.0
    normalized = rms_normalize_iq(x)
    output = derive_features(normalized, "F7")
    expected_i = torch.full_like(output[:, 0, :], 2.0**0.5)
    assert torch.allclose(output[:, 0, :], expected_i)
    assert torch.allclose(output[:, 1, :], torch.zeros_like(output[:, 1, :]))
    assert torch.allclose(output[:, 2, :], expected_i)
    assert torch.allclose(output[:, 3, :], torch.ones_like(output[:, 3, :]))
    assert torch.allclose(output[:, 4, :], torch.zeros_like(output[:, 4, :]))
    assert torch.allclose(output[:, 5:, :], torch.zeros_like(output[:, 5:, :]), atol=1.0e-5)
