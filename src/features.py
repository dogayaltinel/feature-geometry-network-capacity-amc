"""Deterministic IQ feature geometries for the controlled study.

Raw samples are expected as ``(B, 2, T)`` with I on channel 0 and Q on
channel 1. RMS normalization is intentionally separate from feature
derivation so the training loop can enforce augmentation-before-derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


FeatureCode = Literal["F2", "F3", "F5", "F7"]
FEATURE_CHANNELS: dict[str, int] = {"F2": 2, "F3": 3, "F5": 5, "F7": 7}
EPSILON = 1.0e-6


@dataclass(frozen=True)
class FeatureSpec:
    code: FeatureCode
    channels: tuple[str, ...]


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "F2": FeatureSpec("F2", ("I", "Q")),
    "F3": FeatureSpec("F3", ("I", "Q", "magnitude")),
    "F5": FeatureSpec("F5", ("I", "Q", "magnitude", "normalized_I", "normalized_Q")),
    "F7": FeatureSpec(
        "F7",
        ("I", "Q", "magnitude", "normalized_I", "normalized_Q", "phase_difference", "amplitude_difference"),
    ),
}


def validate_feature_code(feature: str) -> FeatureCode:
    if feature not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature code {feature!r}; expected one of {sorted(FEATURE_SPECS)}")
    return feature  # type: ignore[return-value]


def validate_iq(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected IQ tensor with shape (B, 2, T), got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError(f"Expected a floating-point tensor, got {x.dtype}")


def rms_normalize_iq(x: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Normalize each IQ example by its RMS over channels and time."""
    validate_iq(x)
    rms = torch.sqrt(x.square().mean(dim=(1, 2), keepdim=True) + epsilon)
    return x / rms


def _circular_difference(values: torch.Tensor) -> torch.Tensor:
    """Return a wrapped first difference with a deterministic zero first sample."""
    difference = values[..., 1:] - values[..., :-1]
    difference = torch.atan2(torch.sin(difference), torch.cos(difference))
    first = torch.zeros_like(values[..., :1])
    return torch.cat((first, difference), dim=-1)


def _first_difference(values: torch.Tensor) -> torch.Tensor:
    """Return an ordinary first difference with a deterministic zero first sample."""
    difference = values[..., 1:] - values[..., :-1]
    first = torch.zeros_like(values[..., :1])
    return torch.cat((first, difference), dim=-1)


def derive_features(x: torch.Tensor, feature: str, epsilon: float = EPSILON) -> torch.Tensor:
    """Derive one of the locked channel representations from normalized IQ.

    Phase difference uses a wrapped first difference. Amplitude difference is
    an ordinary first difference. Both use a zero first sample. This explicit
    convention is part of the feature contract and must remain synchronized
    with the manifest.
    """
    code = validate_feature_code(feature)
    validate_iq(x)
    i = x[:, 0:1]
    q = x[:, 1:2]
    magnitude = torch.sqrt(i.square() + q.square() + epsilon)
    normalized_i = i / magnitude.clamp_min(epsilon)
    normalized_q = q / magnitude.clamp_min(epsilon)
    channels = [i, q]
    if code in {"F3", "F5", "F7"}:
        channels.append(magnitude)
    if code in {"F5", "F7"}:
        channels.extend((normalized_i, normalized_q))
    if code == "F7":
        phase = torch.atan2(q, i)
        channels.extend((_circular_difference(phase), _first_difference(magnitude)))
    output = torch.cat(channels, dim=1)
    expected = FEATURE_CHANNELS[code]
    if output.shape[1] != expected or not bool(torch.isfinite(output).all()):
        raise FloatingPointError(f"Invalid {code} feature output: shape={tuple(output.shape)}")
    return output


def feature_channel_count(feature: str) -> int:
    return FEATURE_CHANNELS[validate_feature_code(feature)]
