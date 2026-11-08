import numpy as np
import pytest

from reclab.metrics import normalized_entropy, paired_interval, ranking


def test_ne_constant():
    assert normalized_entropy([0, 0, 1, 1], [0.5] * 4) == pytest.approx(1)
    assert normalized_entropy([0, 1], [0.01, 0.99]) < 0.02


@pytest.mark.parametrize(
    "labels,probs",
    [([1, 1], [0.5, 0.5]), ([0, 1], [float("nan"), 0.5]), ([0, 1], [0.2, 2]), ([], [])],
)
def test_invalid_ne(labels, probs):
    with pytest.raises(ValueError):
        normalized_entropy(labels, probs)


def test_full_catalog_seen_mask_and_ties():
    scores = np.array([[99, 3, 3, 1], [1, 2, 3, 4]])
    held = np.array([[0, 1, 5, 1], [0, 3, 4, 2], [1, 2, 2, 2]])
    seen = np.array([[0, 0, 5, 0]])
    result = ranking(scores, held, seen, k=1)
    assert result["recall@1"] == 0.5
    assert result["ndcg@1"] == 1
    assert result["evaluated_users"] == 1
    assert scores[0, 0] == 99


def test_bootstrap_identity():
    a = {"user_ids": [0, 1], "user_recall": [0.5, 1]}
    result = paired_interval(a, a)
    assert result["lower"] == result["upper"] == 0
