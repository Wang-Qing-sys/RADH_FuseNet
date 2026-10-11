"""Seedable tensor corruption and missingness augmentation in normalized units."""
from __future__ import annotations

import torch
from torch import Tensor


def _shift(x: Tensor, mask: Tensor, channel: int, shift: int) -> None:
    """In-place non-circular shift; introduced boundaries are missing."""
    old_x, old_mask = x[:, channel].clone(), mask[:, channel].clone()
    x[:, channel].zero_()
    mask[:, channel].zero_()
    if shift > 0:
        x[:, channel, shift:] = old_x[:, :-shift]
        mask[:, channel, shift:] = old_mask[:, :-shift]
    elif shift < 0:
        x[:, channel, :shift] = old_x[:, -shift:]
        mask[:, channel, :shift] = old_mask[:, -shift:]
    else:
        x[:, channel], mask[:, channel] = old_x, old_mask


def corrupt_view(x: Tensor, mask: Tensor, severity: float = 0.5,
                 kind: str = "mixed", humidity_enabled: bool = True,
                 generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Return a copy with Gaussian/spike/dropout/drift/shift/humidity artifacts.

    Magnitudes are explicit engineering policies, not recovered paper parameters.
    ``mixed`` applies all families. Missingness is always propagated in the mask.
    """
    if kind not in {"mixed", "gaussian", "spikes", "dropout", "drift", "shift", "humidity", "clean"}:
        raise ValueError(f"Unknown corruption: {kind}")
    if severity < 0:
        raise ValueError("severity must be nonnegative.")
    output, observed = x.clone(), mask.bool().clone()
    if severity == 0 or kind == "clean":
        return output, observed
    b, m, length = x.shape
    device = x.device
    rand = lambda shape: torch.rand(shape, device=device, generator=generator)
    randn = lambda shape: torch.randn(shape, device=device, generator=generator)
    if kind in {"mixed", "gaussian"}:
        output += .25 * severity * randn(x.shape)
    if kind in {"mixed", "spikes"}:
        spikes = rand(x.shape) < min(.03 * severity, .3)
        output += spikes * (2.5 * severity * randn(x.shape))
    if kind in {"mixed", "dropout"}:
        width = min(length, max(1, int(length * .15 * severity)))
        for sample in range(b):
            channel = int(torch.randint(m, (), device=device, generator=generator).item())
            start = int(torch.randint(max(1, length-width+1), (), device=device, generator=generator).item())
            observed[sample, channel, start:start+width] = False
    if kind in {"mixed", "drift"}:
        ramp = torch.linspace(-1, 1, length, device=device)
        output += .5 * severity * randn((b, m, 1)) * ramp
    if kind in {"mixed", "shift"}:
        maximum = min(length-1, max(1, int(length * .04 * severity)))
        channel = int(torch.randint(min(m, 3), (), device=device, generator=generator).item())
        shift = int(torch.randint(-maximum, maximum+1, (), device=device, generator=generator).item())
        if shift == 0:
            shift = maximum
        _shift(output, observed, channel, shift)
    if kind in {"mixed", "humidity"} and humidity_enabled and m >= 5:
        ramp = torch.linspace(0, 1, length, device=device)
        ambient = severity * (.5 * randn((b, 1)) + .3 * ramp)
        output[:, 3] += ambient
        output[:, 4] += .6 * ambient
    return torch.where(observed, output, 0), observed


def quality_views(x: Tensor, mask: Tensor, humidity_enabled: bool = True,
                  generator: torch.Generator | None = None) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Nested corruption views: the strong view adds degradation to the mild view."""
    mild = corrupt_view(x, mask, .35, humidity_enabled=humidity_enabled, generator=generator)
    strong = corrupt_view(mild[0], mild[1], 1.0, humidity_enabled=humidity_enabled, generator=generator)
    return mild, strong


def modality_dropout(x: Tensor, mask: Tensor, probability: float = .2,
                     generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    if not 0 <= probability <= 1:
        raise ValueError("modality dropout probability must lie in [0,1].")
    drop = torch.rand(x.shape[:2], device=x.device, generator=generator) < probability
    # Keep at least one originally observed physiological source per example.
    available = mask[:, :3].any(dim=-1)
    for sample in range(len(x)):
        choices = torch.nonzero(available[sample], as_tuple=False).flatten()
        if len(choices) and drop[sample, choices].all():
            drop[sample, choices[0]] = False
    observed = mask & ~drop.unsqueeze(-1)
    return torch.where(observed, x, 0), observed


def span_mask(x: Tensor, mask: Tensor, fraction: float = .15, span_length: int = 8,
              generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Mask random contiguous spans; return input, input mask, held-out target mask."""
    if not 0 < fraction < 1 or span_length < 1:
        raise ValueError("Mask fraction must be (0,1), and span_length positive.")
    observed = mask.clone()
    length = x.shape[-1]
    width = min(span_length, length)
    count = max(1, int(round(length*fraction/width)))
    hidden = torch.zeros_like(mask)
    for sample in range(len(x)):
        for channel in range(x.shape[1]):
            for _ in range(count):
                start = int(torch.randint(max(1, length-width+1), (), device=x.device, generator=generator).item())
                hidden[sample, channel, start:start+width] = True
    hidden &= mask
    observed &= ~hidden
    return torch.where(observed, x, 0), observed, hidden
