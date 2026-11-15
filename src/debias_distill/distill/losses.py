"""Distillation losses on truncated teacher targets.

A cached target keeps, per position, the teacher's log-probability on its top-k
tokens, on the watch-list tokens, and the mass of everything else. The loss is the
KL divergence from that target to the student over the same partition of the
vocabulary: one bucket per kept token and one bucket for the rest.
"""

import torch
import torch.nn.functional as F

MIN_MASS = 1e-8
MAX_KEPT = 1.0 - 1e-6  # float32 cannot represent 1 - 1e-8, and log(1 - 1) is -inf


def truncated_kl(
    student_logits: torch.Tensor,
    top_ids: torch.Tensor,
    top_logp: torch.Tensor,
    watch_ids: torch.Tensor,
    watch_logp: torch.Tensor,
    k: int | None = None,
    use_watch: bool = True,
) -> torch.Tensor:
    """Mean KL(teacher || student) over positions.

    student_logits [..., vocab]; top_ids / top_logp [..., K] sorted by descending
    probability; watch_ids [W]; watch_logp [..., W]. `k` keeps only the first k of the
    top tokens and `use_watch=False` drops the watch-list; the mass they carried moves
    to the rest bucket (the truncation ablation).
    """
    ids, logp = top_ids[..., :k], top_logp[..., :k]
    if use_watch:
        ids = torch.cat([ids, watch_ids.expand(*ids.shape[:-1], -1)], dim=-1)
        logp = torch.cat([logp, watch_logp], dim=-1)
    teacher = logp.float().exp()
    teacher_rest = (1.0 - teacher.sum(-1)).clamp_min(MIN_MASS)
    total = teacher.sum(-1) + teacher_rest  # 1 up to the rounding of the stored values
    teacher, teacher_rest = teacher / total[..., None], teacher_rest / total

    student_logp = student_logits.float().log_softmax(-1).gather(-1, ids)
    kept = student_logp.logsumexp(-1).exp().clamp(max=MAX_KEPT)
    student_rest = torch.log1p(-kept)

    kl = (teacher * (teacher.clamp_min(1e-30).log() - student_logp)).sum(-1)
    kl = kl + teacher_rest * (teacher_rest.log() - student_rest)
    return kl.mean()


def next_token_loss(student_logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of each next token (hard labels, no teacher)."""
    return F.cross_entropy(student_logits[:, :-1].flatten(0, 1).float(), tokens[:, 1:].flatten())
