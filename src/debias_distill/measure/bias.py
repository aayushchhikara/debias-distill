"""Template-based bias metrics.

Primary metric (pre-registered): the mean over (template, subject pair, occupation) of
|log P(occupation | male version) - log P(occupation | female version)|, where the
occupation is scored over all its tokens, article included. Subject pairs are he/she
plus the split's name pairs.

Also computed from the same scores:
  - the score excluding the article, and on the split's unseen occupations only;
  - a noise floor: the same score between two names of the same gender;
  - the mean signed log-ratio per occupation group (positive = likelier after a male subject);
  - paired-template KL: the symmetric KL between the next-token distributions that follow
    the male and the female version of a template.

Intervals are bootstrapped over template stems, because templates that share a stem
differ only in their context prefix and are not independent samples.
"""

from dataclasses import dataclass

import numpy as np

from debias_distill.data.templates import (
    FinalUnlock,
    fill,
    load_occupation_table,
    load_stems,
    load_templates,
    subject_pairs,
)
from debias_distill.measure.scoring import next_token_logprobs, token_logprobs

N_BOOTSTRAP = 10_000


@dataclass
class TemplateScores:
    """Occupation log-probabilities for every (template, subject, occupation) of a split."""

    split: str
    templates: list[str]
    stems: list[str]
    subjects: list[str]  # male and female subject of pair i are at 2i and 2i + 1
    occupations: list[tuple[str, str, str]]  # (occupation, group, first split)
    logp: np.ndarray  # [template, subject, occupation], all occupation tokens
    logp_no_article: np.ndarray  # same, without the article's tokens
    template_kl: np.ndarray  # [template, pair], symmetric KL of the next-token distributions

    @property
    def n_pairs(self) -> int:
        return len(self.subjects) // 2

    def log_ratios(self, no_article: bool = False) -> np.ndarray:
        """log P(occupation | male) - log P(occupation | female): [template, pair, occupation]."""
        logp = self.logp_no_article if no_article else self.logp
        return logp[:, 0::2] - logp[:, 1::2]

    def same_gender_log_ratios(self) -> np.ndarray:
        """Log-ratios between each name and the next name of the same gender.

        [template, 2 * name pairs, occupation]; empty if the split has fewer than two name pairs.
        """
        names = self.logp[:, 2:]  # drop the pronoun pair
        n = names.shape[1] // 2
        if n < 2:
            return np.empty((len(self.templates), 0, len(self.occupations)))
        male, female = names[:, 0::2], names[:, 1::2]
        return np.concatenate(
            [male - np.roll(male, -1, axis=1), female - np.roll(female, -1, axis=1)], axis=1
        )


def score_templates(
    model,
    tokenizer,
    split: str,
    device,
    root: str = "data/templates",
    unlock: FinalUnlock | None = None,
    batch_size: int = 256,
    progress: bool = False,
) -> TemplateScores:
    """Score every occupation after every template and subject of `split`."""
    templates = load_templates(split, root, unlock)
    stems = load_stems(split, root, unlock)
    subjects = [s for pair in subject_pairs(split, root, unlock) for s in pair]
    occupations = load_occupation_table(split, root, unlock)

    def ids(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    bos = tokenizer.eos_token_id  # documents in the corpus are separated by this token
    prompts = [[bos, *ids(fill(t, s))] for t in templates for s in subjects]
    continuations = [ids(" " + occupation) for occupation, _, _ in occupations]
    article_len = [len(ids(" " + occupation.split(" ", 1)[0])) for occupation, _, _ in occupations]

    sequences = [p + c for p in prompts for c in continuations]
    scored = token_logprobs(
        model, sequences, device, batch_size, desc=f"{split} templates" if progress else None
    )
    shape = (len(templates), len(subjects), len(occupations))
    logp, logp_no_article = np.empty(shape), np.empty(shape)
    for n, values in enumerate(scored):
        o = n % len(occupations)
        tail = values[-len(continuations[o]) :]
        index = np.unravel_index(n, shape)
        logp[index] = tail.sum()
        logp_no_article[index] = tail[article_len[o] :].sum()

    nxt = next_token_logprobs(model, prompts, device).astype(np.float64)
    nxt = nxt.reshape(len(templates), len(subjects), -1)
    male, female = nxt[:, 0::2], nxt[:, 1::2]
    kl = 0.5 * (
        (np.exp(male) * (male - female)).sum(-1) + (np.exp(female) * (female - male)).sum(-1)
    )
    return TemplateScores(split, templates, stems, subjects, occupations, logp, logp_no_article, kl)


def stem_bootstrap(
    values: np.ndarray, stems: list[str], n: int = N_BOOTSTRAP, seed: int = 0
) -> np.ndarray:
    """Means of `values` under `n` resamples of the template stems.

    `values` has templates on its first axis. With the same stems and seed the
    resamples are identical across calls, so two models can be compared pairwise.
    """
    names = sorted(set(stems))
    member = np.array([names.index(s) for s in stems])
    per_template = values.reshape(len(stems), -1)
    sums = np.zeros(len(names))
    counts = np.zeros(len(names))
    np.add.at(sums, member, per_template.sum(1))
    np.add.at(counts, member, per_template.shape[1])
    draws = np.random.default_rng(seed).integers(0, len(names), size=(n, len(names)))
    return sums[draws].sum(1) / counts[draws].sum(1)


def with_interval(
    values: np.ndarray, stems: list[str], n: int = N_BOOTSTRAP, seed: int = 0
) -> dict | None:
    """Mean of `values` with a 95% interval bootstrapped over stems; None if empty."""
    if values.size == 0:
        return None
    low, high = np.percentile(stem_bootstrap(values, stems, n, seed), [2.5, 97.5])
    return {"value": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def summarise(scores: TemplateScores, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """All bias metrics for one model on one split."""
    ratios = scores.log_ratios()
    stems = scores.stems
    groups = np.array([group for _, group, _ in scores.occupations])
    unseen = np.array([first == scores.split for _, _, first in scores.occupations])
    if scores.split == "train":
        unseen[:] = False  # every train occupation is visible to training

    def interval(values: np.ndarray) -> dict | None:
        return with_interval(values, stems, n_bootstrap, seed)

    return {
        "mean_abs_log_ratio": interval(np.abs(ratios)),
        "excluding_article": interval(np.abs(scores.log_ratios(no_article=True))),
        "unseen_occupations": interval(np.abs(ratios[:, :, unseen])),
        "seen_occupations": interval(np.abs(ratios[:, :, ~unseen])),
        "pronoun_pair": interval(np.abs(ratios[:, :1])),
        "name_pairs": interval(np.abs(ratios[:, 1:])),
        "noise_floor": interval(np.abs(scores.same_gender_log_ratios())),
        "signed_by_group": {
            str(g): float(ratios[:, :, groups == g].mean()) for g in sorted(set(groups))
        },
        "template_kl": interval(scores.template_kl),
        "counts": {
            "templates": len(scores.templates),
            "stems": len(set(stems)),
            "subject_pairs": scores.n_pairs,
            "occupations": len(scores.occupations),
            "unseen_occupations": int(unseen.sum()),
        },
    }
