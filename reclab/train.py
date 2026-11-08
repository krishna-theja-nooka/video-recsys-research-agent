"""CPU reference training with fixed epochs and paired random seeds."""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from reclab.metrics import evaluate
from reclab.model import TwoTower


@dataclass(frozen=True)
class Config:
    epochs: int = 8
    dim: int = 32
    lr: float = 0.003
    batch_size: int = 1024
    threads: int = 2
    k: int = 10

    def __post_init__(self):
        if not 1 <= self.epochs <= 50 or self.dim not in (16, 32, 64):
            raise ValueError("Training exceeds supported bounds")
        if not 0 < self.lr <= 0.1 or not 16 <= self.batch_size <= 8192:
            raise ValueError("Invalid optimizer parameters")
        if not 1 <= self.threads <= 8 or not 1 <= self.k <= 100:
            raise ValueError("Invalid threads or K")


def fit(train, genres, seed, config=Config(), weights=None, content=True):
    torch.set_num_threads(config.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = TwoTower(int(train[:, 0].max()) + 1, len(genres), genres, config.dim, content)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0001)
    weights = np.ones(len(train)) if weights is None else np.asarray(weights)
    if weights.shape != (len(train),) or not np.isfinite(weights).all():
        raise ValueError("Invalid training weights")
    if (weights < 0.8 - 1e-7).any() or (weights > 1.2 + 1e-7).any():
        raise ValueError("Training weights exceed 20% bound")
    rows = torch.tensor(train, dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float32)
    losses = []
    start = time.perf_counter()
    cpu_start = time.process_time()
    steps = 0
    model.train()
    for _ in range(config.epochs):
        total = 0.0
        for idx in np.array_split(
            rng.permutation(len(rows)), int(np.ceil(len(rows) / config.batch_size))
        ):
            batch = rows[idx]
            logits = model(batch[:, 0], batch[:, 1])
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, (batch[:, 2] >= 4).float(), reduction="none"
                )
                * w[idx]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
            steps += 1
        losses.append(total / len(rows))
    cost = {
        "train_wall_seconds": time.perf_counter() - start,
        "train_cpu_seconds": time.process_time() - cpu_start,
        "optimizer_steps": steps,
        "training_examples": len(rows) * config.epochs,
        "parameters": sum(p.numel() for p in model.parameters()),
        "losses": losses,
        "config": asdict(config),
    }
    return model, cost


def assess(model, heldout, seen, k):
    scores = model.scores()
    logits = np.clip(scores[heldout[:, 0], heldout[:, 1]], -50, 50)
    return evaluate(scores, 1 / (1 + np.exp(-logits)), heldout, seen, k)


def popularity(train, heldout, seen, items, k):
    users = int(train[:, 0].max()) + 1
    counts = np.bincount(train[:, 1], minlength=items)
    positive = np.bincount(train[train[:, 2] >= 4, 1], minlength=items)
    prior = np.mean(train[:, 2] >= 4)
    probs = (positive + prior * 20) / (counts + 20)
    return evaluate(np.tile(positive, (users, 1)), probs[heldout[:, 1]], heldout, seen, k)
