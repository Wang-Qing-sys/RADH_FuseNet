"""Explicit, mask-aware reference objectives for RADH-FuseNet.

Equation (19) specifies L1 reconstruction and shared/private cross-Gram
orthogonality. Equation (40) lists objective families but does not give their
functional forms or weights. The choices below are consequently reference
implementations, not recovered author hyperparameters: expected Dirichlet
cross-entropy + annealed incorrect-evidence KL; Student-t NLL + NIG evidence
regularization; symmetric cross-source InfoNCE; clean/mild/strong hinge ranking;
reliability-to-correctness BCE; and explicit temporal penalties.

Every missing supervision source is masked. ``RADHLoss.last_status`` records
why a term was inactive; a returned zero for absent labels/negatives is not a
claim that this term was optimized. All-empty batches remain differentiable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .evidential import _float, _safe_nig, nig_predictive, nig_predictive_variance


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Mean only selected finite values; an empty selection is differentiable zero."""
    mask = mask.to(device=value.device, dtype=torch.bool) & torch.isfinite(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def dirichlet_kl(alpha: Tensor) -> Tensor:
    """KL(Dir(alpha) || Dir(1)) for each leading index."""
    alpha = _float(alpha).clamp_min(1e-6)
    strength = alpha.sum(-1, keepdim=True)
    c = alpha.new_tensor(float(alpha.shape[-1]))
    log_normalizer = torch.lgamma(strength).squeeze(-1) - torch.lgamma(alpha).sum(-1) - torch.lgamma(c)
    expectation = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength))).sum(-1)
    return (log_normalizer + expectation).clamp_min(0)


def evidential_classification_loss(alpha: Tensor, labels: Tensor, annealing: float = 1.0,
                                   reduction: str = "mean") -> Tensor:
    """Digamma expected CE + annealed KL after removing true-class evidence.

    Negative labels are unlabelled. The target-class concentration is set to one
    only inside KL, so this term penalizes evidence for incorrect classes.
    """
    alpha = _float(alpha).clamp_min(1e-6)
    labels = labels.to(device=alpha.device, dtype=torch.long)
    if labels.shape != alpha.shape[:-1]:
        raise ValueError("Labels must match alpha's leading dimensions.")
    if (labels >= alpha.shape[-1]).any():
        raise ValueError("Class label exceeds the number of classes.")
    valid = labels >= 0
    one_hot = F.one_hot(labels.clamp_min(0), alpha.shape[-1]).to(alpha.dtype)
    expected_ce = (one_hot * (torch.digamma(alpha.sum(-1, keepdim=True)) - torch.digamma(alpha))).sum(-1)
    alpha_without_correct_evidence = one_hot + (1 - one_hot) * alpha
    loss = expected_ce + max(0.0, min(float(annealing), 1.0)) * dirichlet_kl(alpha_without_correct_evidence)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")
    return loss.sum() / valid.sum().clamp_min(1)


def normal_inverse_gamma_nll(nig: Mapping[str, Tensor], target: Tensor) -> Tensor:
    """Elementwise Student-t marginal negative log likelihood of the NIG head.

    The last Student-t parameter in manuscript Eq. (39) is interpreted as scale
    squared: beta*(1+nu)/(nu*alpha). This is the standard NIG marginalization.
    """
    nu, alpha, beta = _safe_nig(nig["nu"], nig["alpha"], nig["beta"])
    residual = _float(target) - _float(nig["mu"])
    log_scale_term = math.log(2 * math.pi) + beta.log() + torch.log1p(nu) - nu.log()
    scaled_error = residual.square() * nu / (2 * beta * (1 + nu))
    return (torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + 0.5 * log_scale_term
            + (alpha + 0.5) * torch.log1p(scaled_error))


def nig_regression_loss(nig: Mapping[str, Tensor], target: Tensor,
                         regularizer: float = 0.01, mask: Tensor | None = None,
                         reduction: str = "mean") -> Tensor:
    """Student-t NLL + lambda*|y-mu|*(2*nu+alpha), ignoring missing targets."""
    target = target.to(device=nig["mu"].device, dtype=nig["mu"].dtype)
    valid = torch.isfinite(target)
    if mask is not None:
        valid = valid & mask.to(device=target.device, dtype=torch.bool)
    # Replace NaNs before arithmetic: masking afterwards can leave NaN gradients.
    safe_target = torch.where(valid, target, nig["mu"].detach())
    nu, alpha, _ = _safe_nig(nig["nu"], nig["alpha"], nig["beta"])
    regularization = (safe_target - nig["mu"]).abs() * (2 * nu + alpha)
    loss = normal_inverse_gamma_nll(nig, safe_target) + float(regularizer) * regularization
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")
    return loss.sum() / valid.sum().clamp_min(1)


nig_loss = nig_regression_loss


def _availability(outputs: Mapping[str, Any], batch: Mapping[str, Tensor]) -> Tensor:
    if "availability" in outputs:
        return outputs["availability"].bool()
    mask = batch.get("target_mask", batch.get("mask"))
    if mask is None:
        return torch.isfinite(batch["x"]).any(-1)
    return mask.bool().any(-1) if mask.ndim == 3 else mask.bool()


def _patch_mask(outputs: Mapping[str, Any], availability: Tensor, length: int) -> Tensor:
    if "patch_availability" in outputs:
        return outputs["patch_availability"].bool()
    return availability.unsqueeze(-1).expand(-1, -1, length)


def shared_private_orthogonality(shared: Tensor, private: Tensor, mask: Tensor) -> Tensor:
    """Normalized Eq. (19) cross-Gram Frobenius penalty, accepting Ds != Dp.

    The temporal cross product is [Ds,Dp]; no elementwise matching of shared
    and private feature dimensions is assumed. Normalization controls its scale
    when patch length or feature widths change.
    """
    if shared.shape[:-1] != private.shape[:-1]:
        raise ValueError("Shared and private representations must have matching batch/source/patch axes.")
    weight = mask.to(shared.dtype).unsqueeze(-1)
    gram = torch.matmul((shared * weight).transpose(-2, -1), private * weight)
    count = weight.sum(-2).clamp_min(1)
    gram = gram / count.unsqueeze(-1)
    return masked_mean(gram.square().mean(dim=(-1, -2)), mask.any(-1))


def physiological_infonce(shared: Tensor, mask: Tensor, temperature: float = 0.1) -> tuple[Tensor, int]:
    """Symmetric InfoNCE between only PPG/PPS/RSP, with batch-instance negatives.

    Paired sources must both be available. A source pair with fewer than two
    paired windows cannot form negatives and is explicitly inactive.
    """
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive.")
    weights = mask.to(shared.dtype).unsqueeze(-1)
    pooled = (shared * weights).sum(-2) / weights.sum(-2).clamp_min(1)
    pooled = F.normalize(pooled, dim=-1)
    available = mask.any(-1)
    losses = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        valid = available[:, left] & available[:, right]
        if int(valid.sum()) < 2:
            continue
        scores = pooled[valid, left] @ pooled[valid, right].transpose(0, 1) / temperature
        labels = torch.arange(scores.shape[0], device=scores.device)
        losses.append(0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)))
    return (torch.stack(losses).mean(), len(losses)) if losses else (shared.sum() * 0, 0)


@dataclass
class LossWeights:
    """Reference weights; tune on participant-disjoint validation, not test data."""
    classification: float = 1.0
    regression: float = 1.0
    reconstruction: float = 0.1
    orthogonality: float = 0.01
    consistency: float = 0.05
    ranking: float = 0.1
    calibration: float = 0.05
    coupling_tv: float = 0.01
    trajectory: float = 0.01
    temporal_smoothness: float = 0.001
    respiration_preservation: float = 0.01


class RADHLoss(nn.Module):
    """Mask-aware two-stage composite objective.

    ``forward(outputs, batch, stage='supervised', epoch=0,
              clean_outputs=..., mild_outputs=..., strong_outputs=...)``
    returns scalar tensors, including ``total``. Pretraining needs a boolean
    ``batch['reconstruction_mask']`` selecting originally observed samples
    hidden from the encoder, and a clean ``reconstruction_target`` in raw input
    units. When provided, model reconstruction_center/scale map this clean
    target to the decoder's standardized coordinates. ``target_mask`` retains
    original sample validity when ``mask`` has been modified by augmentation.
    Wholly unavailable augmented sources have no observed normalization frame
    and are excluded from reconstruction; hidden spans in partly observed
    sources remain supervised. Targets use the model's same +/-50 guard.

    Algorithm 2's pretraining stage optimizes only L_rec + lambda_rank*L_rank;
    the complete weighted objective is enabled during supervised fine-tuning.

    ``y=-1`` and nonfinite ``target`` entries are unlabelled. Ranking acts on all
    five source reliabilities; correctness calibration and contrastive alignment
    use physiological sources and, for calibration, the global head only.
    """

    def __init__(self, weights: Mapping[str, float] | LossWeights | None = None,
                 kl_anneal_epochs: int = 10, ranking_margin: float = 0.1,
                 temperature: float = 0.1, nig_regularizer: float = 0.01) -> None:
        super().__init__()
        self.weights = asdict(LossWeights())
        if isinstance(weights, LossWeights):
            weights = asdict(weights)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
            self.weights.update({k: float(v) for k, v in weights.items()})
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Loss weights must be nonnegative.")
        self.kl_anneal_epochs = max(1, int(kl_anneal_epochs))
        self.ranking_margin = float(ranking_margin)
        self.temperature = float(temperature)
        self.nig_regularizer = float(nig_regularizer)
        self.last_status: dict[str, str] = {}

    def forward(self, outputs: Mapping[str, Any], batch: Mapping[str, Tensor],
                stage: str = "supervised", epoch: int = 0,
                clean_outputs: Mapping[str, Any] | None = None,
                mild_outputs: Mapping[str, Any] | None = None,
                strong_outputs: Mapping[str, Any] | None = None,
                corrupt_outputs: Mapping[str, Any] | tuple[Mapping[str, Any], ...] | None = None
                ) -> dict[str, Tensor]:
        if stage not in {"supervised", "finetune", "pretrain", "pretraining", "ssl"}:
            raise ValueError("stage must be supervised/finetune or pretrain/pretraining/ssl.")
        pretrain = stage in {"pretrain", "pretraining", "ssl"}
        anchor = outputs["alpha"] if "alpha" in outputs else outputs["reconstruction"]
        zero = anchor.sum() * 0
        terms = {key: zero for key in self.weights}
        status = {key: "inactive: required output or supervision absent" for key in self.weights}
        availability = _availability(outputs, batch)
        physiological_available = availability[:, :3].any(-1)

        labels = batch.get("y")
        if not pretrain and labels is not None and "alpha" in outputs:
            valid = (labels >= 0) & physiological_available
            target_labels = torch.where(valid, labels, torch.full_like(labels, -1))
            anneal = min(1.0, max(0.0, float(epoch) / self.kl_anneal_epochs))
            terms["classification"] = evidential_classification_loss(outputs["alpha"], target_labels, anneal)
            status["classification"] = "active" if valid.any() else "inactive: no labels on available physiological inputs"

        regression_target = batch.get("target")
        if not pretrain and regression_target is not None and "nig" in outputs:
            valid = torch.isfinite(regression_target) & physiological_available.unsqueeze(-1)
            terms["regression"] = nig_regression_loss(outputs["nig"], regression_target,
                                                       self.nig_regularizer, valid)
            status["regression"] = "active" if valid.any() else "inactive: no finite regression targets"

        if "reconstruction" in outputs:
            reconstructed = outputs["reconstruction"]
            raw_target = batch.get("reconstruction_target")
            if raw_target is not None and clean_outputs is not None and "corrected_input" in clean_outputs:
                raw_target = clean_outputs["corrected_input"]
            if raw_target is not None and "reconstruction_center" in outputs:
                target = (raw_target - outputs["reconstruction_center"]) / outputs["reconstruction_scale"].clamp_min(1e-6)
                target = target.clamp(-50, 50)
            elif raw_target is not None and clean_outputs is not None and "reconstruction_target" in clean_outputs:
                target = clean_outputs["reconstruction_target"]
            elif raw_target is not None:
                target = raw_target  # For decoders whose target is already in input units.
            else:
                target = outputs.get("reconstruction_target", batch["x"])
            target = target.detach()
            original_mask = batch.get("target_mask", batch.get("mask", torch.isfinite(target))).bool()
            if original_mask.ndim == 2:
                original_mask = original_mask.unsqueeze(-1).expand_as(target)
            valid = original_mask & torch.isfinite(target) & availability.unsqueeze(-1)
            if "reconstruction_mask" in batch:
                valid = valid & batch["reconstruction_mask"].bool()
            elif pretrain:
                raise ValueError("Masked pretraining requires batch['reconstruction_mask'] for known-but-hidden targets.")
            safe_target = torch.where(valid, target, reconstructed.detach())
            terms["reconstruction"] = masked_mean((reconstructed - safe_target).abs(), valid)
            status["reconstruction"] = "active: masked MAE" if valid.any() else "inactive: no reconstruction targets selected"

        if "shared" in outputs and "private" in outputs:
            shared, private = outputs["shared"], outputs["private"]
            patch_mask = _patch_mask(outputs, availability, shared.shape[-2])
            terms["orthogonality"] = shared_private_orthogonality(shared, private, patch_mask)
            status["orthogonality"] = "active" if patch_mask.any() else "inactive: all sources missing"
            consistent_outputs = clean_outputs if clean_outputs is not None else outputs
            consistent_shared = consistent_outputs["shared"]
            consistent_availability = _availability(consistent_outputs, batch)
            consistent_mask = _patch_mask(consistent_outputs, consistent_availability, consistent_shared.shape[-2])
            terms["consistency"], pairs = physiological_infonce(consistent_shared, consistent_mask, self.temperature)
            status["consistency"] = f"active: {pairs} physiological pairs" if pairs else "inactive: fewer than two paired windows for negatives"

        if corrupt_outputs is not None:
            if isinstance(corrupt_outputs, (tuple, list)):
                if len(corrupt_outputs) == 2:
                    mild_outputs, strong_outputs = corrupt_outputs
                elif len(corrupt_outputs) == 1:
                    strong_outputs = corrupt_outputs[0]
                else:
                    raise ValueError("corrupt_outputs must contain one or two views.")
            else:
                mild_outputs = corrupt_outputs.get("mild", mild_outputs)
                strong_outputs = corrupt_outputs.get("strong", strong_outputs)
        clean_view = clean_outputs if clean_outputs is not None else outputs
        views = [view for view in (clean_view, mild_outputs, strong_outputs) if view is not None]
        if len(views) >= 2 and all("reliability" in view for view in views):
            clean_available = _availability(clean_view, batch)
            ranking_losses = []
            for better, worse in zip(views[:-1], views[1:]):
                ranking = F.relu(self.ranking_margin - better["reliability"] + worse["reliability"])
                valid = clean_available[:, :ranking.shape[1]]
                ranking_losses.append(masked_mean(ranking, valid))
            terms["ranking"] = torch.stack(ranking_losses).mean()
            status["ranking"] = "active" if clean_available.any() else "inactive: no originally available source"
        else:
            status["ranking"] = "inactive: requires clean plus mild/strong forward views"

        if not pretrain and labels is not None and "evidence_heads" in outputs:
            evidence = outputs["evidence_heads"]
            head_reliability = outputs.get("head_reliability")
            if head_reliability is None and "reliability" in outputs:
                r = outputs["reliability"][:, :3]
                head_reliability = torch.cat((r, r.mean(-1, keepdim=True)), dim=1)
            if head_reliability is not None:
                head_available = torch.cat((availability[:, :3], physiological_available[:, None]), dim=1)
                valid = head_available & (labels >= 0).unsqueeze(-1)
                correctness = (evidence.argmax(-1) == labels.unsqueeze(-1)).to(evidence.dtype).detach()
                calibrated = F.binary_cross_entropy(head_reliability.clamp(1e-6, 1 - 1e-6), correctness, reduction="none")
                terms["calibration"] = masked_mean(calibrated, valid)
                status["calibration"] = "active" if valid.any() else "inactive: no labelled available affective head"

        sample_mask = outputs.get("sample_mask", batch.get("mask"))
        if sample_mask is not None and sample_mask.ndim == 2:
            sample_mask = sample_mask.unsqueeze(-1).expand_as(batch["x"])
        if "compensation" in outputs and sample_mask is not None:
            coupling = outputs["compensation"]
            if coupling.ndim == 3:
                coupling = coupling.squeeze(1)
            valid_humidity = sample_mask[:, 2] & sample_mask[:, 3] & sample_mask[:, 4]
            adjacent = valid_humidity[:, 1:] & valid_humidity[:, :-1]
            terms["coupling_tv"] = masked_mean(coupling.diff(dim=-1).abs(), adjacent)
            status["coupling_tv"] = "active" if adjacent.any() else "inactive: no adjacent complete humidity samples"
            if "corrected_input" in outputs:
                # Preserve the direction of local respiratory oscillations while
                # allowing baseline/amplitude changes caused by compensation.
                # This normalized first-difference penalty is an explicit choice;
                # the manuscript specifies preservation but no formula or band.
                reference = outputs.get("raw_input", batch["x"])[:, 4].detach()
                corrected = outputs["corrected_input"][:, 2]
                d_ref, d_corr = reference.diff(dim=-1), corrected.diff(dim=-1)
                d_ref = torch.where(adjacent, d_ref, torch.zeros_like(d_ref))
                d_corr = torch.where(adjacent, d_corr, torch.zeros_like(d_corr))
                ref_norm = d_ref.square().sum(-1).clamp_min(1e-12).sqrt()
                corr_norm = d_corr.square().sum(-1).clamp_min(1e-12).sqrt()
                eligible = adjacent.any(-1) & (ref_norm > 1e-6) & (corr_norm > 1e-6)
                correlation = (d_ref * d_corr).sum(-1) / (ref_norm * corr_norm).clamp_min(1e-6)
                terms["respiration_preservation"] = masked_mean(1 - correlation.clamp(-1, 1), eligible)
                status["respiration_preservation"] = "active" if eligible.any() else "inactive: no nonconstant paired respiration samples"

        if not pretrain and "nig" in outputs:
            mu = outputs["nig"]["mu"]
            delta = mu.diff(dim=-1)
            available_steps = physiological_available.unsqueeze(-1).expand_as(delta)
            terms["temporal_smoothness"] = masked_mean(delta.abs(), available_steps)
            status["temporal_smoothness"] = "active" if available_steps.any() and delta.numel() else "inactive: no multi-step available forecasts"
            if regression_target is not None:
                valid_target = torch.isfinite(regression_target)
                safe_target = torch.where(valid_target, regression_target, torch.zeros_like(regression_target))
                valid_pairs = valid_target[:, 1:] & valid_target[:, :-1] & available_steps
                terms["trajectory"] = masked_mean((delta - safe_target.diff(dim=-1)).abs(), valid_pairs)
                status["trajectory"] = "active" if valid_pairs.any() else "inactive: no adjacent labelled forecast steps"

        total = zero
        for name, weight in self.weights.items():
            if pretrain and name not in {"reconstruction", "ranking"}:
                terms[name] = zero
                status[name] = "inactive: Algorithm 2 stage A uses reconstruction and ranking only"
                continue
            if pretrain and name == "reconstruction" and weight:
                weight = 1.0
            if weight:
                total = total + weight * terms[name]
            else:
                status[name] = "disabled: configured weight is zero"
        self.last_status = status
        return {"total": total, **terms}


def compute_losses(outputs: Mapping[str, Any], batch: Mapping[str, Tensor],
                    weights: Mapping[str, float] | LossWeights | None = None,
                    **kwargs: Any) -> dict[str, Tensor]:
    """Convenience wrapper; keep a RADHLoss instance when inspecting last_status."""
    return RADHLoss(weights=weights)(outputs, batch, **kwargs)
