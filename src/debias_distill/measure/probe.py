"""Linear probes: can the subject's gender be read from the hidden state after a template?

For every layer, a logistic-regression probe is fitted on the hidden state at the last
token of each filled template (just before the occupation). Folds hold out template
stems and subject pairs together, so a probe is always tested on templates and subjects
it has not seen. The control repeats the fit with shuffled training labels; its accuracy
should stay near 0.5.

Exploratory: not a success criterion.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from debias_distill.data.templates import (
    FinalUnlock,
    fill,
    load_stems,
    load_templates,
    subject_pairs,
)
from debias_distill.measure.scoring import final_hidden_states

N_FOLDS = 4


def fold_masks(stem_of: np.ndarray, pair_of: np.ndarray, n_folds: int = N_FOLDS, seed: int = 0):
    """Yield (train, test) masks; a fold's test stems and test pairs never appear in its training set."""
    rng = np.random.default_rng(seed)
    stem_fold = dict(zip(rng.permutation(np.unique(stem_of)).tolist(), _cycle(n_folds)))
    pair_fold = dict(zip(rng.permutation(np.unique(pair_of)).tolist(), _cycle(n_folds)))
    s = np.array([stem_fold[x] for x in stem_of.tolist()])
    p = np.array([pair_fold[x] for x in pair_of.tolist()])
    for k in range(n_folds):
        train, test = (s != k) & (p != k), (s == k) & (p == k)
        if train.any() and test.any():
            yield train, test


def _cycle(n: int):
    i = 0
    while True:
        yield i % n
        i += 1


def probe_features(
    features: np.ndarray,
    labels: np.ndarray,
    stem_of: np.ndarray,
    pair_of: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> dict:
    """Held-out probe accuracy and shuffled-label control for one layer's features [n, width]."""
    rng = np.random.default_rng(seed)
    correct = control = total = 0
    for train, test in fold_masks(stem_of, pair_of, n_folds, seed):
        scaler = StandardScaler().fit(features[train])
        x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])
        for shuffled in (False, True):
            y = rng.permutation(labels[train]) if shuffled else labels[train]
            predicted = LogisticRegression(max_iter=2000).fit(x_train, y).predict(x_test)
            n_hits = int((predicted == labels[test]).sum())
            if shuffled:
                control += n_hits
            else:
                correct += n_hits
        total += int(test.sum())
    return {"accuracy": correct / total, "control_accuracy": control / total, "n_test": total}


def probe_accuracy(
    model,
    tokenizer,
    split: str,
    device,
    root: str = "data/templates",
    unlock: FinalUnlock | None = None,
    seed: int = 0,
) -> dict:
    """Per-layer probe accuracy on `split`. Layer 0 is the embedding output."""
    templates = load_templates(split, root, unlock)
    stems = load_stems(split, root, unlock)
    pairs = subject_pairs(split, root, unlock)
    bos = tokenizer.eos_token_id
    sequences, labels, stem_of, pair_of = [], [], [], []
    for template, stem in zip(templates, stems):
        for p, pair in enumerate(pairs):
            for gender, subject in enumerate(pair):
                text = fill(template, subject)
                sequences.append([bos, *tokenizer(text, add_special_tokens=False)["input_ids"]])
                labels.append(gender)
                stem_of.append(stem)
                pair_of.append(p)
    states = final_hidden_states(model, sequences, device)
    labels, stem_of, pair_of = np.array(labels), np.array(stem_of), np.array(pair_of)
    layers = [
        probe_features(states[:, layer], labels, stem_of, pair_of, seed=seed)
        for layer in range(states.shape[1])
    ]
    return {
        "layers": layers,
        "max_accuracy": max(layer["accuracy"] for layer in layers),
        "mean_control_accuracy": float(np.mean([layer["control_accuracy"] for layer in layers])),
    }
