"""Exact full-catalog ranking and pointwise rating-label metrics."""

import numpy as np


def normalized_entropy(labels, probabilities):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching nonempty binary labels and predictions")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite in [0, 1]")
    base = y.mean()
    if base in (0, 1):
        raise ValueError("Normalized entropy is undefined for a single-class partition")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()
    entropy = -(base * np.log(base) + (1 - base) * np.log1p(-base))
    return float(loss / entropy)


def ranking(scores, heldout, seen, k=10):
    scores = np.array(scores, dtype=float, copy=True)
    if not 1 <= k <= scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("Invalid scores or K")
    scores[seen[:, 0], seen[:, 1]] = -np.inf
    positives = heldout[heldout[:, 2] >= 4]
    values, ndcg, user_ids = [], [], []
    for u in np.unique(positives[:, 0]):
        relevant = set(positives[positives[:, 0] == u, 1].tolist())
        # Stable tie-breaking uses mapped item ID; seen items never enter results.
        candidates = np.flatnonzero(np.isfinite(scores[u]))
        top = candidates[np.argsort(-scores[u, candidates], kind="stable")[:k]]
        hits = np.array([int(i in relevant) for i in top])
        values.append(float(hits.sum() / len(relevant)))
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
        ndcg.append(float(np.sum(hits / np.log2(np.arange(len(top)) + 2)) / ideal))
        user_ids.append(int(u))
    if not values:
        raise ValueError("No positive evaluation users")
    return {
        f"recall@{k}": float(np.mean(values)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        "evaluated_users": len(values),
        "user_ids": user_ids,
        "user_recall": values,
    }


def evaluate(scores, probabilities, heldout, seen, k=10):
    result = ranking(scores, heldout, seen, k)
    result["normalized_entropy"] = normalized_entropy(heldout[:, 2] >= 4, probabilities)
    return result


def paired_interval(baseline, candidate, seed=42, samples=2000):
    if baseline["user_ids"] != candidate["user_ids"]:
        raise ValueError("Paired bootstrap requires identical users")
    diff = np.array(candidate["user_recall"]) - np.array(baseline["user_recall"])
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, (samples, len(diff)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(diff.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": samples,
    }
