"""Distillation losses."""

import torch


def topk_kl_loss(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_probs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between cached teacher targets and student predictions."""
    raise NotImplementedError


def total_loss(kd: torch.Tensor, ce: torch.Tensor | None, hard_label_weight: float) -> torch.Tensor:
    """Combine distillation and optional hard-label losses."""
    raise NotImplementedError
