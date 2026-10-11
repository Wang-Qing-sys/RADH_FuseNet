"""RADH-FuseNet: executable, manuscript-grounded PyTorch reference model.

Input channel order is always PPG, PPS, RSP, EHS, RHS. Environmental channels
have their own encoders but no affect classifier. They influence compensation,
quality and relational context only. A window without physiological input
returns a uniform class distribution and total uncertainty, even if humidity
channels remain present.

This is an offline-window reference, not author-provided training code or an
exact numerical reproduction. Table 2 dimensions/topology are respected;
unreported SQI thresholds, parameterizations and sparse incidence details are
explicit implementation choices. The SSM is an actual selective diagonal
state-space recurrence, implemented in portable PyTorch rather than a fused
Mamba kernel. Upstream synchronization, antialias resampling and sensor-specific
wavelet filtering must be performed before calling this model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import CrossScaleTemporalFusion, DynamicHypergraphLayer, SourceEncoder
from .evidential import fuse_evidence


SOURCE_NAMES = ("PPG", "PPS", "RSP", "EHS", "RHS")


@dataclass
class ModelConfig:
    num_classes: int = 5
    horizon: int = 1
    hidden_dim: int = 96
    shared_dim: int = 64
    private_dim: int = 32
    patch_size: int = 32
    ssm_depth: int = 4
    state_dim: int = 16
    ssm_expansion: int = 2
    hypergraph_layers: int = 2
    top_k: int = 2
    temporal_depths: tuple[int, int, int] = (1, 2, 6)
    dropout: float = 0.1
    humidity_compensation: bool = True
    conflict_threshold: float = 0.4
    sampling_rate: float = 32.0
    n_fft: int = 64
    reliability_ema: float = 0.8
    prior_groups: tuple[tuple[int, ...], ...] | None = None
    quality_bands: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        for name in ("num_classes", "horizon", "hidden_dim", "shared_dim",
                     "private_dim", "patch_size", "ssm_depth", "state_dim",
                     "ssm_expansion", "hypergraph_layers", "top_k", "n_fft"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.prior_groups is not None:
            self.prior_groups = tuple(tuple(group) for group in self.prior_groups)
            if not self.prior_groups or any(not group or any(
                    not isinstance(source, int) or isinstance(source, bool) or not 0 <= source < 5
                    for source in group) or len(set(group)) != len(group)
                    for group in self.prior_groups):
                raise ValueError("prior_groups must be nonempty groups of unique source indices 0..4")
        if self.quality_bands is not None:
            self.quality_bands = tuple(tuple(band) for band in self.quality_bands)
            if len(self.quality_bands) != 5 or any(
                    len(band) != 2 or not all(math.isfinite(float(v)) for v in band)
                    or not 0 <= band[0] < band[1] for band in self.quality_bands):
                raise ValueError("quality_bands must contain five finite (low, high) pairs with 0 <= low < high")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.n_fft < 4:
            raise ValueError("n_fft must be at least 4")
        self.temporal_depths = tuple(self.temporal_depths)
        if len(self.temporal_depths) != 3 or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 1
                for d in self.temporal_depths):
            raise ValueError("temporal_depths must contain three positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0 < self.conflict_threshold < 1:
            raise ValueError("conflict_threshold must lie in (0, 1)")
        if not 0 <= self.reliability_ema < 1:
            raise ValueError("reliability_ema must lie in [0, 1)")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")


def masked_quantile(x: Tensor, mask: Tensor, quantile: float) -> Tensor:
    """Nearest-rank quantile over the final axis; absent rows safely return zero."""
    count = mask.sum(-1)
    sorted_values = x.masked_fill(~mask, float("inf")).sort(dim=-1).values
    index = ((count - 1).clamp_min(0).to(x.dtype) * quantile).long()
    result = sorted_values.gather(-1, index[..., None]).squeeze(-1)
    return torch.where(count > 0, result, torch.zeros_like(result))


def robust_standardize(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Observed-sample median/MAD standardization; missing samples stay exactly 0."""
    median = masked_quantile(x, mask, 0.5)
    deviation = torch.where(mask, x - median[..., None], torch.zeros_like(x))
    mad = masked_quantile(deviation.abs(), mask, 0.5)
    # RMS fallback handles low-duty-cycle signals whose exact MAD is zero.
    rms = (deviation.square().sum(-1) / mask.sum(-1).clamp_min(1)).clamp_min(1e-12).sqrt()
    scale = torch.where(mad > 1e-5, 1.4826 * mad, rms).clamp_min(1e-4)
    # No observed amplitudes imply no estimable normalization. Use the neutral
    # transform, never a tiny divisor that inflates a clean reconstruction target.
    scale = torch.where(mask.any(-1), scale, torch.ones_like(scale))
    normalized = torch.where(mask, deviation / scale[..., None], torch.zeros_like(x))
    # A numerical guard, not sensor saturation clipping. Large spikes remain large.
    return normalized.clamp(-50, 50), median, scale


class HumidityCompensation(nn.Module):
    """Bounded local ambient coupling from RHS/EHS and their first differences.

    Equation (6) is interpreted literally: RSP_corrected = RHS - beta * EHS.
    We apply it only where RSP, EHS and RHS are all observed. Otherwise the
    supplied observed RSP is retained. Humidity channels never manufacture an
    available physiological modality when its explicit RSP mask is false.
    This missing-humidity fallback is a documented reference assumption.
    """

    def __init__(self) -> None:
        super().__init__()
        self.coupling = nn.Sequential(nn.Linear(8, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized, _, _ = robust_standardize(x[:, 3:5], mask[:, 3:5])
        ehs, rhs = normalized[:, 0], normalized[:, 1]
        ehs_diff = F.pad(ehs[:, 1:] - ehs[:, :-1], (1, 0))
        rhs_diff = F.pad(rhs[:, 1:] - rhs[:, :-1], (1, 0))
        # Differences across gaps are not actual measured increments.
        ehs_pair = F.pad(mask[:, 3, 1:] & mask[:, 3, :-1], (1, 0), value=False)
        rhs_pair = F.pad(mask[:, 4, 1:] & mask[:, 4, :-1], (1, 0), value=False)
        ehs_diff = ehs_diff * ehs_pair
        rhs_diff = rhs_diff * rhs_pair
        ranges = masked_quantile(normalized, mask[:, 3:5], 0.9) - masked_quantile(
            normalized, mask[:, 3:5], 0.1)
        energy = (normalized.square() * mask[:, 3:5]).sum(-1)
        energy = energy / mask[:, 3:5].sum(-1).clamp_min(1)
        stats = torch.cat((ranges, energy.log1p()), dim=-1)
        stats = stats[:, None].expand(-1, x.shape[-1], -1)
        local = torch.stack((rhs, ehs, rhs_diff, ehs_diff), dim=-1)
        beta = torch.sigmoid(self.coupling(torch.cat((local, stats), -1)).squeeze(-1))
        valid = mask[:, 2] & mask[:, 3] & mask[:, 4]
        beta = beta * valid.to(beta.dtype)
        corrected_rsp = torch.where(valid, x[:, 4] - beta * x[:, 3], x[:, 2])
        corrected = torch.cat((x[:, :2], corrected_rsp[:, None], x[:, 3:]), dim=1)
        return corrected, beta, valid


class SignalQualityEstimator(nn.Module):
    """Thirteen interpretable, window-level SQI descriptors and source MLPs.

    Descriptors implement the manuscript's amplitude, energy, kurtosis,
    concentration, jump, saturation, missingness, autocorrelation, band energy,
    peak-interval regularity, contour continuity, drift and humidity-coupling
    cues. Thresholds and expected bands are illustrative sensor defaults, not
    fitted physiological validity criteria. Calibration on the actual sensors
    and sampling rate is required before deployment.
    """

    descriptor_names = (
        "robust_amplitude_range", "log_energy", "kurtosis", "spectral_concentration",
        "jump_rate", "saturation_ratio", "missing_ratio", "autocorrelation",
        "expected_band_energy", "peak_interval_regularity", "contour_continuity",
        "baseline_drift", "residual_humidity_coupling",
    )

    def __init__(self, sampling_rate: float, bands: tuple[tuple[float, float], ...] | None = None) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.bands = bands or ((0.5, 4.0), (0.5, 4.0), (0.05, 1.0), (0.0, 0.5), (0.05, 1.0))
        self.networks = nn.ModuleList([
            nn.Sequential(nn.Linear(13, 24), nn.GELU(), nn.Linear(24, 1)) for _ in range(5)
        ])

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, modalities, length = x.shape
        observed = mask.to(x.dtype)
        count = observed.sum(-1).clamp_min(1)
        availability = observed.mean(-1)
        mean = (x * observed).sum(-1) / count
        centered = torch.where(mask, x - mean[:, :, None], torch.zeros_like(x))
        energy = centered.square().sum(-1) / count
        kurtosis = (centered.pow(4).sum(-1) / count) / energy.square().clamp_min(1e-6)
        amplitude = masked_quantile(x, mask, 0.95) - masked_quantile(x, mask, 0.05)
        if length > 1:
            adjacent = mask[:, :, 1:] & mask[:, :, :-1]
            pair_count = adjacent.sum(-1).clamp_min(1)
            difference = (x[:, :, 1:] - x[:, :, :-1]) * adjacent
            jump_rate = ((difference.abs() > 3.0) & adjacent).sum(-1) / pair_count
            contour = torch.exp(-difference.abs().sum(-1) / pair_count)
            autocorr = (centered[:, :, 1:] * centered[:, :, :-1] * adjacent).sum(-1)
            autocorr = autocorr / (pair_count * energy).clamp_min(1e-6)
        else:
            jump_rate = torch.zeros_like(mean)
            contour = torch.zeros_like(mean)
            autocorr = torch.zeros_like(mean)
        maximum = x.masked_fill(~mask, -float("inf")).amax(-1)
        minimum = x.masked_fill(~mask, float("inf")).amin(-1)
        boundary = ((x == maximum[:, :, None]) | (x == minimum[:, :, None])) & mask
        saturation = boundary.sum(-1).to(x.dtype) / count
        # A rectangular observed window is used here; missingness is a separate cue.
        spectrum = torch.fft.rfft(centered.float(), dim=-1).abs().square()
        if spectrum.shape[-1] > 1:
            spectrum = spectrum.clone()
            spectrum[..., 0] = 0
        spectral_sum = spectrum.sum(-1).clamp_min(1e-6)
        concentration = spectrum.amax(-1) / spectral_sum
        frequency = torch.fft.rfftfreq(length, d=1.0 / self.sampling_rate, device=x.device)
        band = torch.stack([((frequency >= low) & (frequency <= high))
                            for low, high in self.bands], dim=0)
        band_energy = (spectrum * band[None]).sum(-1) / spectral_sum
        peak_regularity = torch.zeros_like(mean)
        if length >= 3:
            peaks = ((x[:, :, 1:-1] > x[:, :, :-2])
                     & (x[:, :, 1:-1] >= x[:, :, 2:])
                     & mask[:, :, :-2] & mask[:, :, 1:-1] & mask[:, :, 2:])
            peaks = F.pad(peaks, (1, 1), value=False)
            positions = torch.arange(length, device=x.device)[None, None].expand_as(x)
            last_peak = torch.where(peaks, positions, -torch.ones_like(positions)).cummax(-1).values
            previous_peak = F.pad(last_peak[:, :, :-1], (1, 0), value=-1)
            intervals_valid = peaks & (previous_peak >= 0)
            intervals = (positions - previous_peak).to(x.dtype) * intervals_valid
            interval_count = intervals_valid.sum(-1)
            average_interval = intervals.sum(-1) / interval_count.clamp_min(1)
            variance = ((intervals - average_interval[:, :, None]).square()
                        * intervals_valid).sum(-1) / interval_count.clamp_min(1)
            regularity = torch.exp(-variance.clamp_min(1e-12).sqrt()
                                   / average_interval.clamp_min(1))
            peak_regularity = regularity * (interval_count >= 2)
        midpoint = max(1, length // 2)
        left = (x[:, :, :midpoint] * observed[:, :, :midpoint]).sum(-1)
        left = left / observed[:, :, :midpoint].sum(-1).clamp_min(1)
        right = (x[:, :, midpoint:] * observed[:, :, midpoint:]).sum(-1)
        right = right / observed[:, :, midpoint:].sum(-1).clamp_min(1)
        drift = (right - left).abs() / energy.clamp_min(1e-6).sqrt()
        humidity_coupling = torch.zeros_like(mean)
        ambient = centered[:, 3]
        for source in (2, 3, 4):
            joint_mask = mask[:, source] & mask[:, 3]
            numerator = (centered[:, source] * ambient * joint_mask).sum(-1).abs()
            denominator = ((centered[:, source].square() * joint_mask).sum(-1)
                           * (ambient.square() * joint_mask).sum(-1)).clamp_min(1e-12).sqrt()
            humidity_coupling[:, source] = numerator / denominator
        descriptors = torch.stack((
            amplitude / 10.0, energy.log1p(), kurtosis.clamp(0, 100) / 10.0,
            concentration, jump_rate, saturation, 1.0 - availability,
            autocorr.clamp(-1, 1), band_energy, peak_regularity, contour,
            drift.clamp(0, 10) / 10.0, humidity_coupling.clamp(0, 1)), dim=-1).to(x.dtype)
        descriptors = torch.nan_to_num(descriptors, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.stack([network(descriptors[:, source]).squeeze(-1)
                              for source, network in enumerate(self.networks)], dim=1)
        # A mild deterministic safeguard limits obvious flatline/jump artifacts;
        # source-specific technical quality is still learned by the MLPs.
        usable_variation = energy / (energy + 0.05)
        artifact_penalty = torch.exp(-2.0 * jump_rate)
        quality = torch.sigmoid(logits) * usable_variation * artifact_penalty
        quality = quality * (availability > 0).to(x.dtype)
        return quality, descriptors


class ReliabilityEstimator(nn.Module):
    """Four-cue sigmoid estimator with monotonic cue directions.

    q, exp(-reconstruction residual), availability, cosine agreement mapped to
    [0,1] have nonnegative learned coefficients. This guarantees cue-wise
    monotonicity, not a guarantee that every possible corruption reduces the
    trained model's reliability. Clean/corrupt ranking loss supplies that signal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5, 4))
        self.bias = nn.Parameter(torch.full((5,), -1.0))

    def forward(self, quality: Tensor, residual: Tensor, availability: Tensor,
                shared: Tensor, patch_availability: Tensor) -> tuple[Tensor, Tensor]:
        weight = patch_availability[:, :, :, None]
        pooled = (shared * weight).sum(2) / weight.sum(2).clamp_min(1e-6)
        source_weight = quality * availability
        total = (pooled * source_weight[:, :, None]).sum(1, keepdim=True)
        others = total - pooled * source_weight[:, :, None]
        others_weight = source_weight.sum(1, keepdim=True) - source_weight
        consensus = others / others_weight[:, :, None].clamp_min(1e-6)
        agreement = F.cosine_similarity(pooled, consensus, dim=-1, eps=1e-6)
        # No peer is a neutral cue, rather than fictitious perfect agreement.
        agreement = torch.where(others_weight > 1e-6, 0.5 * (agreement + 1),
                                torch.full_like(agreement, 0.5))
        cues = torch.stack((quality, torch.exp(-residual.clamp(0, 20)),
                            availability, agreement), dim=-1)
        reliability = torch.sigmoid((cues * F.softplus(self.weight)[None]).sum(-1)
                                    + self.bias[None])
        # Exact zeros are essential to missing-source propagation and ignorance.
        reliability = reliability * availability * quality
        return reliability, cues


class RADHFuseNet(nn.Module):
    """Five-source reliability-aware dynamic-hypergraph evidential model.

    ``mask`` is boolean [B,5,T], True for actually observed samples. Nonfinite
    samples are automatically marked missing. Unobserved stored values cannot
    influence any output. ``previous_reliability`` is optional [B,5] and is only
    suitable for adjacent windows of the SAME subjects in the SAME batch order.
    There is deliberately no hidden, cross-batch EMA state.

    Reconstruction is in corrected, robust-standardized coordinates, with its
    target supplied as ``reconstruction_target``. ``corrected_input`` retains
    physical input units. Shared/private outputs are patch-level factors.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.compensator = HumidityCompensation()
        self.quality_estimator = SignalQualityEstimator(cfg.sampling_rate, cfg.quality_bands)
        self.encoders = nn.ModuleList([
            SourceEncoder(cfg.hidden_dim, cfg.shared_dim, cfg.private_dim,
                          cfg.ssm_depth, cfg.state_dim, cfg.ssm_expansion,
                          cfg.n_fft, cfg.patch_size, cfg.dropout) for _ in SOURCE_NAMES
        ])
        self.reliability_estimator = ReliabilityEstimator()
        self.hypergraph = nn.ModuleList([
            DynamicHypergraphLayer(cfg.shared_dim, cfg.top_k, cfg.dropout, cfg.prior_groups)
            for _ in range(cfg.hypergraph_layers)
        ])
        self.temporal_projection = nn.Linear(cfg.shared_dim, cfg.hidden_dim)
        self.temporal = CrossScaleTemporalFusion(cfg.hidden_dim, cfg.temporal_depths,
                                                 cfg.state_dim, cfg.ssm_expansion,
                                                 cfg.dropout)
        self.physiological_heads = nn.ModuleList([
            nn.Linear(cfg.shared_dim, cfg.num_classes) for _ in range(3)
        ])
        self.global_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.regression_head = nn.Linear(cfg.hidden_dim, 4 * cfg.horizon)

    def forward(self, x: Tensor, mask: Tensor | None = None,
                previous_reliability: Tensor | None = None) -> dict[str, Any]:
        if x.ndim != 3 or x.shape[1] != 5 or x.shape[-1] < 1 or x.shape[0] < 1:
            raise ValueError("x must have shape [B, 5, T], with B and T positive")
        if not x.is_floating_point():
            x = x.float()
        if mask is None:
            mask = torch.isfinite(x)
        else:
            if mask.shape != x.shape or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean, have shape [B, 5, T], and use True=observed")
            mask = mask.to(device=x.device) & torch.isfinite(x)
        x = torch.where(mask, x, torch.zeros_like(x))
        batch, _, length = x.shape
        if self.config.humidity_compensation:
            corrected, compensation, compensation_mask = self.compensator(x, mask)
        else:
            corrected = x
            compensation = x.new_zeros(batch, length)
            compensation_mask = torch.zeros_like(mask[:, 2])
        corrected = torch.where(mask, corrected, torch.zeros_like(corrected))
        normalized, center, scale = robust_standardize(corrected, mask)
        availability = mask.to(x.dtype).mean(-1)
        patch_mask = F.pad(mask.to(x.dtype), (0, (-length) % self.config.patch_size))
        patch_availability = patch_mask.reshape(batch, 5, -1, self.config.patch_size).mean(-1)
        quality, quality_descriptors = self.quality_estimator(normalized, mask)
        branches = [encoder.encode_branches(normalized[:, source], mask[:, source], quality[:, source])
                    for source, encoder in enumerate(self.encoders)]
        provisional = [encoder.fuse(branches[source], mask[:, source], quality[:, source])
                       for source, encoder in enumerate(self.encoders)]
        provisional_shared = torch.stack([item[0] for item in provisional], dim=1)
        provisional_reconstruction = torch.stack([item[2] for item in provisional], dim=1)
        residual = ((provisional_reconstruction - normalized).square() * mask).sum(-1)
        residual = residual / mask.sum(-1).clamp_min(1)
        reliability, reliability_cues = self.reliability_estimator(
            quality, residual, availability, provisional_shared, patch_availability)
        if previous_reliability is not None:
            if previous_reliability.shape != reliability.shape:
                raise ValueError("previous_reliability must have shape [B, 5]")
            previous = previous_reliability.to(reliability).detach()
            if not torch.isfinite(previous).all():
                raise ValueError("previous_reliability must contain finite values")
            # Smooth rises, react immediately to sudden degradation or disappearance.
            ema = self.config.reliability_ema * previous.clamp(0, 1)
            ema = ema + (1.0 - self.config.reliability_ema) * reliability
            reliability = torch.minimum(reliability, ema) * (availability > 0)
        # A two-pass gate avoids a circular dependence: the provisional factors
        # estimate reliability, which then gates the final three-branch encoding.
        representations = [encoder.fuse(branches[source], mask[:, source], reliability[:, source])
                           for source, encoder in enumerate(self.encoders)]
        shared = torch.stack([item[0] for item in representations], dim=1)
        private = torch.stack([item[1] for item in representations], dim=1)
        reconstruction = torch.stack([item[2] for item in representations], dim=1)
        branch_weights = torch.stack([item[3] for item in representations], dim=1)
        fused_nodes = shared
        for layer in self.hypergraph:
            fused_nodes = layer(fused_nodes, reliability, patch_availability)
        # Context can influence physiological nodes through explicit relations;
        # only physiological nodes enter this global affective readout.
        phys_weight = reliability[:, :3, None] * patch_availability[:, :3]
        global_tokens = (fused_nodes[:, :3] * phys_weight[:, :, :, None]).sum(1)
        global_tokens = global_tokens / phys_weight.sum(1)[:, :, None].clamp_min(1e-6)
        global_reliability = reliability[:, :3].mean(-1)
        temporal_features, temporal_weights = self.temporal(
            self.temporal_projection(global_tokens), global_reliability)
        token_weight = patch_availability[:, :3].amax(1)
        global_feature = (temporal_features * token_weight[:, :, None]).sum(1)
        global_feature = global_feature / token_weight.sum(1)[:, None].clamp_min(1e-6)
        source_features = (fused_nodes[:, :3] * patch_availability[:, :3, :, None]).sum(2)
        source_features = source_features / patch_availability[:, :3].sum(2)[:, :, None].clamp_min(1e-6)
        source_evidence = torch.stack([
            F.softplus(head(source_features[:, source]))
            for source, head in enumerate(self.physiological_heads)], dim=1)
        global_evidence = F.softplus(self.global_head(global_feature))
        evidence_heads = torch.cat((source_evidence, global_evidence[:, None]), dim=1)
        head_reliability = torch.cat((reliability[:, :3], global_reliability[:, None]), dim=1)
        fused_opinion = fuse_evidence(evidence_heads, head_reliability,
                                      threshold=self.config.conflict_threshold)
        nig_raw = self.regression_head(global_feature).reshape(batch, self.config.horizon, 4)
        mu = nig_raw[..., 0]
        nu = F.softplus(nig_raw[..., 1]) + 1e-4
        nig_alpha = F.softplus(nig_raw[..., 2]) + 1.0 + 1e-4
        beta = F.softplus(nig_raw[..., 3]) + 1e-4
        predictive_variance = beta * (1.0 + nu) / (nu * (nig_alpha - 1.0))
        output: dict[str, Any] = dict(fused_opinion)
        output.update({
            "probabilities": fused_opinion["probabilities"],
            "probs": fused_opinion["probabilities"],
            "evidence_heads": evidence_heads,
            "source_evidence": source_evidence,
            "head_reliability": head_reliability,
            "reliability": reliability,
            "quality": quality,
            "quality_descriptors": quality_descriptors,
            "availability": availability,
            "patch_availability": patch_availability,
            "sample_mask": mask,
            "raw_input": x,
            "corrected_input": corrected,
            "reconstruction_target": normalized,
            "normalization_center": center,
            "normalization_scale": scale,
            "reconstruction_center": center[:, :, None],
            "reconstruction_scale": scale[:, :, None],
            "reconstruction": reconstruction,
            "shared": shared,
            "private": private,
            "fused_nodes": fused_nodes,
            "compensation": compensation,
            "compensation_mask": compensation_mask,
            "reconstruction_residual": residual,
            "reliability_cues": reliability_cues,
            "branch_weights": branch_weights,
            "temporal_features": temporal_features,
            "temporal_weights": temporal_weights,
            "nig": {"mu": mu, "nu": nu, "alpha": nig_alpha, "beta": beta},
            "predictive_variance": predictive_variance,
            "trajectory_valid": (availability[:, :3].sum(-1) > 0),
        })
        return output
