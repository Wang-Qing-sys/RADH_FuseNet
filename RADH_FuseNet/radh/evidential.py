from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def _float(tensor: Tensor) -> Tensor:
    """Evaluate special functions/fusion in at least float32 under AMP."""
    return tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.float()


def _normalize_opinion(belief: Tensor, uncertainty: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    belief = _float(belief).clamp_min(0)
    uncertainty = _float(uncertainty).clamp_min(0)
    total = belief.sum(-1) + uncertainty
    valid = total > eps
    belief = belief / total.clamp_min(eps).unsqueeze(-1)
    uncertainty = torch.where(valid, uncertainty / total.clamp_min(eps), torch.ones_like(total))
    uncertainty = uncertainty.clamp(min=eps, max=1.0)
    mass = belief.sum(-1)
    # Enforce sum(b) + u = 1 after numerical floors, including vacuous opinions.
    has_belief = mass > eps
    uncertainty = torch.where(has_belief, uncertainty, torch.ones_like(uncertainty))
    safe_mass = torch.where(has_belief, mass, torch.ones_like(mass))
    belief = belief * ((1.0 - uncertainty) / safe_mass).unsqueeze(-1)
    return belief, uncertainty


def evidence_to_opinion(evidence: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Convert nonnegative evidence [..., C] into (belief [..., C], u [...]).

    Non-finite evidence is rejected: silently treating a failed network as a
    reliable prediction would hide a training error. Finite negative evidence
    is clipped to zero; model heads should normally use softplus instead.
    """
    if evidence.ndim < 1 or evidence.shape[-1] < 2:
        raise ValueError("Evidence requires a final class dimension C >= 2.")
    if not torch.isfinite(evidence).all():
        raise ValueError("Evidence must be finite.")
    evidence = _float(evidence).clamp(min=0.0, max=1e12)
    classes = evidence.shape[-1]
    strength = evidence.sum(-1) + classes
    return _normalize_opinion(evidence / strength.unsqueeze(-1), classes / strength, eps)


def discount_opinion(belief: Tensor, uncertainty: Tensor, reliability: Tensor,
                     eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Discount belief into ignorance. r=0 is exactly the vacuous opinion."""
    belief, uncertainty = _normalize_opinion(belief, uncertainty, eps)
    reliability = _float(reliability).clamp(0, 1)
    if reliability.shape != uncertainty.shape:
        raise ValueError("Reliability must match the opinion's non-class dimensions.")
    if not torch.isfinite(reliability).all():
        raise ValueError("Reliability must be finite.")
    return _normalize_opinion(belief * reliability.unsqueeze(-1),
                              1 - reliability + reliability * uncertainty, eps)


def combine_opinions(belief_a: Tensor, uncertainty_a: Tensor,
                     belief_b: Tensor, uncertainty_b: Tensor,
                     threshold: float = 0.4, eps: float = 1e-8) -> dict[str, Tensor]:
    """Combine two opinions using Dempster at low and Yager at high conflict.

    K = sum_{i != j} b_ai b_bj.
    b_num = b_a b_b + b_a u_b + b_b u_a; u_num = u_a u_b.
    Dempster: (b_num, u_num)/(1-K).
    Yager: (b_num, u_num+K).

    The threshold is a piecewise differentiable rule; the discrete choice and
    reliability sort themselves do not have useful derivatives. All selected
    belief/mass computations retain gradients.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Conflict threshold must lie in [0, 1].")
    if belief_a.shape != belief_b.shape or uncertainty_a.shape != uncertainty_b.shape:
        raise ValueError("Both opinions must have the same shapes.")
    a, ua = _normalize_opinion(belief_a, uncertainty_a, eps)
    b, ub = _normalize_opinion(belief_b, uncertainty_b, eps)
    conflict = (a.sum(-1) * b.sum(-1) - (a * b).sum(-1)).clamp(0, 1)
    numerator = a * b + a * ub.unsqueeze(-1) + b * ua.unsqueeze(-1)
    unknown = ua * ub
    # Algebraically 1-K, but this form avoids catastrophic cancellation.
    denominator = (numerator.sum(-1) + unknown).clamp_min(eps)
    dempster_b = numerator / denominator.unsqueeze(-1)
    dempster_u = unknown / denominator
    high_conflict = conflict >= threshold
    fused_b = torch.where(high_conflict.unsqueeze(-1), numerator, dempster_b)
    fused_u = torch.where(high_conflict, unknown + conflict, dempster_u)
    fused_b, fused_u = _normalize_opinion(fused_b, fused_u, eps)
    return {"belief": fused_b, "uncertainty": fused_u, "conflict": conflict,
            "used_yager": high_conflict}


def opinion_to_dirichlet(belief: Tensor, uncertainty: Tensor,
                          eps: float = 1e-8) -> Tensor:
    """Equivalent Dirichlet alpha, using the uniform base rate (prior = 1)."""
    belief, uncertainty = _normalize_opinion(belief, uncertainty, eps)
    return 1 + belief * (belief.shape[-1] / uncertainty).unsqueeze(-1)


def fuse_evidence(evidence: Tensor, reliability: Tensor, threshold: float = 0.4,
                  eps: float = 1e-8) -> dict[str, Tensor]:
    """Fuse evidence [B, 4, C] and reliability [B, 4].

    Head order before sorting is PPG/PPS/RSP/global. ``conflict`` is the maximum
    of the three sequential pairwise conflicts, not an independent probability
    of prediction error. Yager combination is not associative, so the stable
    descending reliability order is part of the model definition. Ties retain
    the declared head order. All-zero reliability yields alpha=1, uniform class
    probabilities, u=1 and K=0, with finite outputs and gradients.
    """
    if evidence.ndim != 3 or evidence.shape[1] != 4:
        raise ValueError("Expected four heads in evidence [B, 4, C]: PPG/PPS/RSP/global.")
    if reliability.shape != evidence.shape[:2]:
        raise ValueError("Expected reliability [B, 4].")
    belief, uncertainty = evidence_to_opinion(evidence, eps)
    belief, uncertainty = discount_opinion(belief, uncertainty, reliability, eps)
    order = torch.argsort(reliability, dim=1, descending=True, stable=True)
    sorted_b = belief.gather(1, order.unsqueeze(-1).expand_as(belief))
    sorted_u = uncertainty.gather(1, order)
    fused_b, fused_u = sorted_b[:, 0], sorted_u[:, 0]
    conflicts, high_flags = [], []
    for head in range(1, 4):
        combined = combine_opinions(fused_b, fused_u, sorted_b[:, head], sorted_u[:, head],
                                    threshold, eps)
        fused_b, fused_u = combined["belief"], combined["uncertainty"]
        conflicts.append(combined["conflict"])
        high_flags.append(combined["used_yager"])
    alpha = opinion_to_dirichlet(fused_b, fused_u, eps)
    step_conflicts = torch.stack(conflicts, dim=1)
    return {"alpha": alpha, "probabilities": alpha / alpha.sum(-1, keepdim=True),
            "uncertainty": fused_u, "conflict": step_conflicts.amax(dim=1),
            "belief": fused_b, "head_belief": belief, "head_uncertainty": uncertainty,
            "fusion_order": order, "step_conflicts": step_conflicts,
            "high_conflict_steps": torch.stack(high_flags, dim=1)}


def _safe_nig(nu: Tensor, alpha: Tensor, beta: Tensor,
              eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    return _float(nu).clamp_min(eps), _float(alpha).clamp_min(1 + eps), _float(beta).clamp_min(eps)


def nig_predictive_variance(nu: Tensor, alpha: Tensor, beta: Tensor,
                            eps: float = 1e-6) -> Tensor:
    """Total NIG predictive variance = beta*(1+nu)/(nu*(alpha-1))."""
    nu, alpha, beta = _safe_nig(nu, alpha, beta, eps)
    return beta / (alpha - 1) * (1 + nu.reciprocal())


def nig_predictive(nig: Mapping[str, Tensor], eps: float = 1e-6) -> dict[str, Tensor]:
    """Student-t prediction plus the aleatoric/epistemic variance decomposition.

    Scale is the Student-t scale parameter, NOT its standard deviation.
    Degrees of freedom = 2*alpha; scale^2 = beta*(1+nu)/(nu*alpha).
    """
    nu, alpha, beta = _safe_nig(nig["nu"], nig["alpha"], nig["beta"], eps)
    aleatoric = beta / (alpha - 1)
    epistemic = aleatoric / nu
    return {"mean": nig["mu"], "df": 2 * alpha,
            "scale": (beta * (1 + nu.reciprocal()) / alpha).sqrt(),
            "aleatoric_variance": aleatoric, "epistemic_variance": epistemic,
            "variance": aleatoric + epistemic}
