import numpy as np
import pytest

from reclab.agent import PROPOSALS, gate, search, weights_for


def test_bounds():
    rows = np.array([[0, 0, 3, 10], [0, 1, 4, 20], [1, 1, 2, 30]])
    copy = rows.copy()
    for name in PROPOSALS:
        assert ((weights_for(rows, name) >= 0.8) & (weights_for(rows, name) <= 1.2)).all()
    assert np.array_equal(rows, copy)
    with pytest.raises(ValueError):
        weights_for(rows, "delete all data")


def test_gate_ne_and_recall():
    base = {"recall@10": 0.1, "normalized_entropy": 1.0}
    assert gate({"recall@10": 0.12, "normalized_entropy": 1.01}, base, 10)
    assert not gate({"recall@10": 0.12, "normalized_entropy": 1.1}, base, 10)
    assert not gate(base, base, 10)


def test_graph_bounded_rejects_all():
    train = np.array([[0, 0, 3, 10], [0, 1, 4, 20]])
    baseline = {"recall@10": 0.2, "normalized_entropy": 1.0}
    calls = []

    def evaluate(name, weights):
        calls.append(name)
        return baseline.copy()

    result = search(train, baseline, evaluate)
    assert result["selected"] == "baseline"
    assert calls == list(PROPOSALS)
    assert len(result["audit"]) == 3


def test_selects_best_qualifying():
    train = np.array([[0, 0, 3, 10], [0, 1, 4, 20]])
    baseline = {"recall@10": 0.2, "normalized_entropy": 1.0}
    scores = dict(recency=0.21, tail=0.23, ambiguous=0.22)
    result = search(
        train, baseline, lambda n, w: {"recall@10": scores[n], "normalized_entropy": 1.0}
    )
    assert result["selected"] == "tail"
