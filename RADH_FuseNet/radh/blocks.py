from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """Left-padded convolution: output at t never reads an input after t."""

    def forward(self, x: Tensor) -> Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (padding, 0)))


class SelectiveSSMBlock(nn.Module):
    """Residual, selective diagonal SSM with an O(L) Python scan.

    A = -exp(A_log), dt_t = softplus(W_dt x_t), and B_t/C_t are
    input-dependent. The discretization is the widely used stable exponential
    transition plus Euler input term:
        h_t = exp(dt_t A) h_(t-1) + dt_t B_t u_t
        y_t = C_t h_t + D u_t.
    The scan is evaluated in float32 for numerical stability under autocast.
    This transparent reference favors portability over fused-kernel throughput.
    """

    def __init__(self, dim: int, state_dim: int = 16, expansion: int = 2,
                 dropout: float = 0.1, conv_kernel: int = 4) -> None:
        super().__init__()
        self.inner_dim = dim * expansion
        self.state_dim = state_dim
        self.dt_rank = max(1, math.ceil(dim / 16))
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.inner_dim)
        self.conv = CausalConv1d(self.inner_dim, self.inner_dim, conv_kernel,
                                 groups=self.inner_dim)
        self.selective = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim,
                                   bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim)
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log()
                                  .repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Slow initial timescales support a useful long-context initialization.
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x: Tensor) -> Tensor:
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2)).transpose(1, 2))
        dt_low, b, c = self.selective(u).split(
            (self.dt_rank, self.state_dim, self.state_dim), dim=-1)
        dt = F.softplus(self.dt_proj(dt_low).float()).clamp(1e-5, 1.0)
        a = -self.A_log.float().exp()
        state = u.new_zeros((u.shape[0], self.inner_dim, self.state_dim),
                            dtype=torch.float32)
        outputs = []
        uf, bf, cf = u.float(), b.float(), c.float()
        for t in range(x.shape[1]):
            step = dt[:, t, :, None]
            state = (torch.exp(step * a[None]) * state
                     + step * bf[:, t, None, :] * uf[:, t, :, None])
            yt = (state * cf[:, t, None, :]).sum(-1) + self.D.float() * uf[:, t]
            outputs.append(yt)
        y = torch.stack(outputs, dim=1).to(u.dtype) * F.silu(gate)
        return x + self.dropout(self.out_proj(y))


class SSMStack(nn.Module):
    def __init__(self, dim: int, depth: int, state_dim: int, expansion: int,
                 dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(dim, state_dim, expansion, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class LocalResidualBlock(nn.Module):
    """Depthwise morphology filters plus a causal low-frequency residual path."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.filters = nn.ModuleList([
            CausalConv1d(dim, dim, k, groups=dim) for k in (3, 7, 15)
        ])
        self.mix = nn.Conv1d(4 * dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x).transpose(1, 2)
        low = F.avg_pool1d(F.pad(z, (2, 0)), kernel_size=3, stride=1)
        z = torch.cat([F.gelu(conv(z)) for conv in self.filters] + [low], dim=1)
        return x + self.dropout(self.mix(z).transpose(1, 2))


class ComplexConv2d(nn.Module):
    """A genuinely complex learned convolution, represented by two real kernels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.real = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.imag = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return (self.real(real) - self.imag(imag),
                self.real(imag) + self.imag(real))


class SpectralBranch(nn.Module):
    """64-point (by default) STFT, quality-conditioned frequency gate, two blocks.

    STFT frames are centered and frequency/frame convolutions are noncausal.
    This branch intentionally analyzes the supplied window, not a live stream.
    """

    def __init__(self, dim: int, n_fft: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = max(1, n_fft // 4)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.frequency_logits = nn.Parameter(torch.zeros(n_fft // 2 + 1))
        self.quality_gate = nn.Linear(1, n_fft // 2 + 1)
        self.blocks = nn.ModuleList([ComplexConv2d(dim) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, quality: Tensor) -> Tensor:
        batch, length, dim = x.shape
        z = self.norm(x).transpose(1, 2).reshape(batch * dim, length)
        # STFT/ISTFT need float32 on CPU and for short arbitrary FFT lengths.
        spec = torch.stft(z.float(), self.n_fft, self.hop_length,
                          window=self.window.float(), center=True,
                          pad_mode="constant", return_complex=True)
        spec = spec.reshape(batch, dim, spec.shape[-2], spec.shape[-1])
        real, imag = spec.real, spec.imag
        for block in self.blocks:
            dr, di = block(real, imag)
            real, imag = real + 0.1 * F.gelu(dr), imag + 0.1 * F.gelu(di)
        gate = torch.sigmoid(self.frequency_logits[None]
                             + self.quality_gate(quality[:, None]))
        spec = torch.complex(real * gate[:, None, :, None],
                             imag * gate[:, None, :, None])
        signal = torch.istft(spec.reshape(batch * dim, *spec.shape[-2:]),
                             self.n_fft, self.hop_length,
                             window=self.window.float(), center=True, length=length)
        signal = signal.reshape(batch, dim, length).transpose(1, 2).to(x.dtype)
        return x + self.dropout(self.mix(signal))


class SourceEncoder(nn.Module):
    """One source-specific encoder; no parameters are shared between sensors."""

    def __init__(self, hidden_dim: int, shared_dim: int, private_dim: int,
                 ssm_depth: int, state_dim: int, expansion: int, n_fft: int,
                 patch_size: int, dropout: float) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stem = CausalConv1d(1, hidden_dim, 3)
        self.stem_norm = nn.LayerNorm(hidden_dim)
        self.local = nn.Sequential(*[
            LocalResidualBlock(hidden_dim, dropout) for _ in range(2)
        ])
        self.spectral = SpectralBranch(hidden_dim, n_fft, dropout)
        self.long = SSMStack(hidden_dim, ssm_depth, state_dim, expansion, dropout)
        self.branch_gate = nn.Sequential(nn.Linear(3 * hidden_dim + 1, hidden_dim),
                                          nn.GELU(), nn.Linear(hidden_dim, 3))
        self.shared_proj = nn.Linear(hidden_dim, shared_dim)
        self.private_proj = nn.Linear(hidden_dim, private_dim)
        self.decoder = nn.Sequential(nn.Linear(shared_dim + private_dim, hidden_dim),
                                      nn.GELU(), nn.Linear(hidden_dim, patch_size))

    def encode_branches(self, signal: Tensor, mask: Tensor,
                        quality: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem_norm(self.stem(signal[:, None]).transpose(1, 2))
        # Mask the stem output too, so convolution bias cannot create observations.
        x = x * mask[:, :, None].to(x.dtype)
        return self.local(x), self.spectral(x, quality), self.long(x)

    def fuse(self, branches: Sequence[Tensor], mask: Tensor,
             reliability: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weight = mask[:, :, None].to(branches[0].dtype)
        count = weight.sum(1).clamp_min(1)
        pooled = [(branch * weight).sum(1) / count for branch in branches]
        gates = self.branch_gate(torch.cat(pooled + [reliability[:, None]], -1))
        gates = gates.softmax(-1)
        fused = sum(branch * gates[:, i, None, None]
                    for i, branch in enumerate(branches))
        batch, length, dim = fused.shape
        padding = (-length) % self.patch_size
        context = F.pad(fused, (0, 0, 0, padding))
        fused = F.pad(fused * weight, (0, 0, 0, padding))
        patch_mask = F.pad(mask.to(fused.dtype), (0, padding))
        patch_count = patch_mask.reshape(batch, -1, self.patch_size).sum(-1)
        nodes = fused.reshape(batch, -1, self.patch_size, dim).sum(2)
        nodes = nodes / patch_count[:, :, None].clamp_min(1)
        # For a fully hidden patch, reconstruct from context already propagated
        # by the three branches, rather than a constant decoder bias. These
        # factors still have zero graph membership/readout weight, as governed
        # by the separate patch availability mask. No hidden input value is read.
        real_positions = fused.new_ones(batch, length)
        real_positions = F.pad(real_positions, (0, padding))
        context_count = real_positions.reshape(batch, -1, self.patch_size).sum(-1)
        context_nodes = context.reshape(batch, -1, self.patch_size, dim).sum(2)
        context_nodes = context_nodes / context_count[:, :, None].clamp_min(1)
        nodes = torch.where(patch_count[:, :, None] > 0, nodes, context_nodes)
        shared = self.shared_proj(nodes)
        private = self.private_proj(nodes)
        reconstruction = self.decoder(torch.cat((shared, private), dim=-1))
        reconstruction = reconstruction.reshape(batch, -1)[:, :length]
        return shared, private, reconstruction, gates


def _scatter_sum(values: Tensor, indices: Tensor, count: int) -> Tensor:
    """Scatter along incidence dimension, preserving optional feature dimension."""
    shape = (values.shape[0], count) + values.shape[2:]
    output = values.new_zeros(shape)
    index = indices
    while index.ndim < values.ndim:
        index = index.unsqueeze(-1)
    return output.scatter_add(1, index.expand_as(values), values)


class DynamicHypergraphLayer(nn.Module):
    """Physiological priors plus neighbor-patch top-k adaptive hyperedges.

    Three prior groups are cardiovascular (PPG/PPS), respiration-context
    (RSP/EHS/RHS), and environment (EHS/RHS); a fourth global group includes all
    available sources at each patch. The exact respiration membership and
    distance-to-incidence map are explicit reference choices where unspecified.
    Adaptive edges contain an anchor and its top-k non-self neighbors in the
    same or adjacent patch. Only O(K * M * 3M) candidates are materialized.
    """

    def __init__(self, dim: int, top_k: int = 2, dropout: float = 0.1,
                 prior_groups: Sequence[Sequence[int]] | None = None) -> None:
        super().__init__()
        self.top_k = top_k
        self.prior_groups = tuple(tuple(group) for group in prior_groups) if prior_groups is not None else (
            (0, 1), (2, 3, 4), (3, 4), (0, 1, 2, 3, 4))
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.update = nn.GRUCell(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.prior_logit = nn.Parameter(torch.tensor(0.0))

    def _incidences(self, x: Tensor, reliability: Tensor,
                    patch_availability: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        batch, modalities, patches, dim = x.shape
        nodes = x.reshape(batch, -1, dim)
        normalized = F.normalize(nodes, dim=-1, eps=1e-6)
        node_rel = (reliability[:, :, None] * patch_availability).reshape(batch, -1)
        device = x.device
        prior_nodes, prior_edges, prior_centers = [], [], []
        groups = self.prior_groups
        for group_id, group in enumerate(groups):
            ids = torch.tensor(group, device=device)[:, None] * patches
            ids = (ids + torch.arange(patches, device=device)[None]).reshape(-1)
            edges = (group_id * patches + torch.arange(patches, device=device))
            edges = edges.repeat(len(group))
            prior_nodes.append(ids)
            prior_edges.append(edges)
            group_features = normalized.reshape(batch, modalities, patches, dim)
            group_weights = node_rel.reshape(batch, modalities, patches)[:, group]
            center = (group_features[:, group] * group_weights[:, :, :, None]).sum(1)
            center = F.normalize(center, dim=-1, eps=1e-6)
            prior_centers.append(center[:, None].expand(-1, len(group), -1, -1)
                                 .reshape(batch, -1, dim))
        pn = torch.cat(prior_nodes)[None].expand(batch, -1)
        pe = torch.cat(prior_edges)[None].expand(batch, -1)
        pc = torch.cat(prior_centers, dim=1)
        member = normalized.gather(1, pn[:, :, None].expand(-1, -1, dim))
        p_distance = 1.0 - (member * pc).sum(-1).clamp(-1, 1)
        p_weight = torch.exp(-p_distance) * node_rel.gather(1, pn)
        p_weight = p_weight * torch.sigmoid(self.prior_logit)

        # Bounded local candidates, rather than all-pairs similarity.
        n_nodes = modalities * patches
        node_id = torch.arange(n_nodes, device=device)
        patch_id = node_id.remainder(patches)
        neighbor_patch = patch_id[:, None, None] + torch.arange(-1, 2, device=device)[None, :, None]
        source_id = torch.arange(modalities, device=device)[None, None, :]
        candidates = neighbor_patch.clamp(0, patches - 1) + source_id * patches
        candidates = candidates.reshape(n_nodes, -1)
        valid = ((neighbor_patch >= 0) & (neighbor_patch < patches))
        valid = valid.expand(-1, -1, modalities).reshape(n_nodes, -1)
        valid = valid & (candidates != node_id[:, None])
        candidate_features = normalized[:, candidates]
        similarities = (normalized[:, :, None] * candidate_features).sum(-1)
        candidate_rel = node_rel[:, candidates]
        valid = valid[None] & (candidate_rel > 0) & (node_rel[:, :, None] > 0)
        scores = similarities.masked_fill(~valid, -1e4)
        top_k = min(self.top_k, candidates.shape[-1])
        indices = scores.topk(top_k, dim=-1).indices
        an = candidates[None].expand(batch, -1, -1).gather(2, indices)
        aw = candidate_rel.gather(2, indices) * torch.exp(
            similarities.gather(2, indices).clamp(-1, 1) - 1)
        aw = aw * valid.gather(2, indices).to(x.dtype)
        # An edge is inactive if its anchor is unavailable.
        aw = aw * (node_rel[:, :, None] > 0).to(x.dtype)
        an = torch.cat((node_id[None, :, None].expand(batch, -1, -1), an), dim=-1)
        aw = torch.cat((node_rel[:, :, None], aw), dim=-1)
        aw = aw * (1.0 - torch.sigmoid(self.prior_logit))
        ae = (len(groups) * patches + node_id)[None, :, None].expand(batch, -1, top_k + 1)
        return (torch.cat((pn, an.reshape(batch, -1)), dim=1),
                torch.cat((pe, ae.reshape(batch, -1)), dim=1),
                torch.cat((p_weight, aw.reshape(batch, -1)), dim=1),
                len(groups) * patches + n_nodes)

    def forward(self, x: Tensor, reliability: Tensor,
                patch_availability: Tensor) -> Tensor:
        batch, modalities, patches, dim = x.shape
        normalized = self.norm(x)
        nodes = normalized.reshape(batch, -1, dim)
        node_ids, edge_ids, incidence, edge_count = self._incidences(
            normalized, reliability, patch_availability)
        members = nodes.gather(1, node_ids[:, :, None].expand(-1, -1, dim))
        edge_degree = _scatter_sum(incidence, edge_ids, edge_count)
        edge_context = _scatter_sum(members * incidence[:, :, None], edge_ids, edge_count)
        edge_context = edge_context / edge_degree[:, :, None].clamp_min(1e-6)
        edge_at_member = edge_context.gather(1, edge_ids[:, :, None].expand(-1, -1, dim))
        score = (self.query(members) * self.key(edge_at_member)).sum(-1) / math.sqrt(dim)
        # Multiplication by incidence implements a log-reliability attention bias,
        # while allowing exact zero for unavailable nodes (no log(0)).
        attention = score.clamp(-12, 12).exp() * incidence
        attention = attention / _scatter_sum(attention, edge_ids, edge_count).gather(
            1, edge_ids).clamp_min(1e-6)
        edges = _scatter_sum(self.value(members) * attention[:, :, None], edge_ids, edge_count)
        node_degree = _scatter_sum(incidence, node_ids, modalities * patches)
        degree = (node_degree.gather(1, node_ids) * edge_degree.gather(1, edge_ids))
        normalized_incidence = incidence / degree.clamp_min(1e-12).sqrt()
        messages = edges.gather(1, edge_ids[:, :, None].expand(-1, -1, dim))
        messages = _scatter_sum(messages * normalized_incidence[:, :, None],
                                node_ids, modalities * patches)
        receive_reliability = (reliability[:, :, None] * patch_availability)
        receive_reliability = receive_reliability.reshape(batch, -1, 1)
        residual = x.reshape(batch, -1, dim)
        proposal = self.update(self.output(messages).reshape(-1, dim),
                               residual.reshape(-1, dim)).reshape(batch, -1, dim)
        output = residual + receive_reliability * self.dropout(proposal - residual)
        present = (patch_availability > 0).reshape(batch, -1, 1).to(output.dtype)
        return (output * present).reshape(batch, modalities, patches, dim)


class CrossScaleTemporalFusion(nn.Module):
    """Causal scale-1/2/4 SSMs with past-only pooling and hold upsampling.

    A coarse token at original position t uses only inputs at positions <= t.
    Repeating that token forward preserves causality. Linear interpolation or
    full-group pooling would leak future tokens and is deliberately avoided.
    The context gate uses cumulative statistics, not a full-sequence mean.
    """

    def __init__(self, dim: int, depths: Sequence[int], state_dim: int,
                 expansion: int, dropout: float) -> None:
        super().__init__()
        self.scales = (1, 2, 4)
        self.paths = nn.ModuleList([
            SSMStack(dim, depth, state_dim, expansion, dropout) for depth in depths
        ])
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in self.scales])
        self.context_gate = nn.Sequential(nn.Linear(3 * dim + 2, dim), nn.GELU(),
                                           nn.Linear(dim, 3))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mean_reliability: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, dim = x.shape
        outputs = []
        for scale, path, projection in zip(self.scales, self.paths, self.projections):
            if scale == 1:
                down = x
            else:
                down = F.avg_pool1d(F.pad(x.transpose(1, 2), (scale - 1, 0)),
                                     kernel_size=scale, stride=scale).transpose(1, 2)
            out = projection(path(down)).repeat_interleave(scale, dim=1)[:, :length]
            outputs.append(out)
        count = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)[None, :, None]
        context = torch.cat([output.cumsum(1) / count for output in outputs], dim=-1)
        variation = F.pad((x[:, 1:] - x[:, :-1]).abs().mean(-1), (1, 0))
        reliability = mean_reliability[:, None].expand(-1, length)
        gates = self.context_gate(torch.cat((context, reliability[:, :, None],
                                             variation[:, :, None]), dim=-1)).softmax(-1)
        fused = sum(output * gates[:, :, i, None] for i, output in enumerate(outputs))
        return self.norm(fused), gates
