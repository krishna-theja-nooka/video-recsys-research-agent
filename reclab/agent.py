"""Deterministic LangGraph controller; no LLM or arbitrary code generation."""

from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from reclab.data import digest

PROPOSALS = ("recency", "tail", "ambiguous")


def weights_for(train, proposal):
    if proposal == "baseline":
        return np.ones(len(train))
    if proposal not in PROPOSALS:
        raise ValueError("Proposal is outside the allowlist")
    if proposal == "recency":
        t = train[:, 3].astype(float)
        return np.clip(0.8 + 0.4 * (t - t.min()) / max(1, np.ptp(t)), 0.8, 1.2)
    if proposal == "tail":
        counts = np.bincount(train[:, 1])
        threshold = np.median(counts[counts > 0])
        return np.where(counts[train[:, 1]] <= threshold, 1.2, 1.0)
    return np.where(train[:, 2] == 3, 0.8, 1.0)


def gate(candidate, baseline, k):
    key = f"recall@{k}"
    return (
        np.isfinite(candidate[key])
        and np.isfinite(candidate["normalized_entropy"])
        and candidate[key] >= baseline[key] + 0.001
        and candidate["normalized_entropy"] <= baseline["normalized_entropy"] * 1.02
    )


class State(TypedDict):
    index: int
    proposal: str
    candidate: dict
    selected: str
    best_recall: float
    audit: list


def search(train, baseline, evaluate_candidate, k=10):
    """Only train and validation evaluator are passed; no test partition argument."""
    frozen_digest = digest(train)

    def propose(state):
        return {"proposal": PROPOSALS[state["index"]]}

    def trial(state):
        if digest(train) != frozen_digest:
            raise ValueError("Baseline data changed")
        weights = weights_for(train, state["proposal"])
        if not ((weights >= 0.8) & (weights <= 1.2)).all():
            raise ValueError("Unbounded proposal")
        result = evaluate_candidate(state["proposal"], weights)
        return {"candidate": result}

    def decide(state):
        candidate = state["candidate"]
        accepted = bool(gate(candidate, baseline, k))
        selected, best = state["selected"], state["best_recall"]
        if accepted and candidate[f"recall@{k}"] > best:
            selected, best = state["proposal"], candidate[f"recall@{k}"]
        entry = {
            "proposal": state["proposal"],
            "passes_baseline_gate": accepted,
            "validation": {
                a: b for a, b in candidate.items() if a not in ("user_ids", "user_recall")
            },
            "training_sha256": frozen_digest,
        }
        return {
            "selected": selected,
            "best_recall": best,
            "audit": state["audit"] + [entry],
            "index": state["index"] + 1,
        }

    graph = StateGraph(State)
    graph.add_node("propose", propose)
    graph.add_node("trial", trial)
    graph.add_node("gate", decide)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "trial")
    graph.add_edge("trial", "gate")
    graph.add_conditional_edges("gate", lambda s: "propose" if s["index"] < 3 else END)
    result = graph.compile().invoke(
        {"index": 0, "selected": "baseline", "best_recall": baseline[f"recall@{k}"], "audit": []},
        config={"recursion_limit": 12},
    )
    if digest(train) != frozen_digest:
        raise ValueError("Training data was modified")
    return result
