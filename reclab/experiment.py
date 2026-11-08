"""Selection on validation only, frozen decisions, then final test evaluation."""

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch

from reclab.agent import search, weights_for
from reclab.data import load_genres, load_partition
from reclab.metrics import paired_interval
from reclab.train import Config, assess, fit, popularity


def compact(metrics):
    return {k: v for k, v in metrics.items() if k not in ("user_ids", "user_recall")}


def log_run(name, seed, cost, metrics, manifest):
    with mlflow.start_run(run_name=f"{name}-seed-{seed}") as active:
        mlflow.log_params(
            {
                **cost["config"],
                "variant": name,
                "seed": seed,
                "split_digest": manifest["partitions"]["train"]["sha256"],
            }
        )
        mlflow.log_metrics(
            {
                "validation/" + k.replace("@", "_at_"): v
                for k, v in compact(metrics).items()
                if isinstance(v, (int, float))
            }
        )
        mlflow.log_metrics({k: v for k, v in cost.items() if isinstance(v, (int, float))})
        for step, loss in enumerate(cost["losses"]):
            mlflow.log_metric("training_loss", loss, step=step)
        return active.info.run_id


def run(data, output, config=Config(), seeds=(42, 43, 44)):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output directory; existing experiments are immutable")
    if len(set(seeds)) != len(seeds) or not 1 <= len(seeds) <= 5:
        raise ValueError("Supply 1..5 unique seeds")
    output.mkdir(parents=True)
    start = time.perf_counter()
    mlflow.set_tracking_uri((output / "mlruns").resolve().as_uri())
    mlflow.set_experiment("video-recsys")
    manifest = json.loads((data / "manifest.json").read_text())
    train = load_partition(data, "train")
    valid = load_partition(data, "valid")
    genres = load_genres(data)
    models, records = {}, []

    def train_variant(name, seed, weights=None, content=True):
        model, cost = fit(train, genres, seed, config, weights, content)
        metrics = assess(model, valid, train, config.k)
        run_id = log_run(name, seed, cost, metrics, manifest)
        models[name, seed] = model
        records.append(
            {
                "variant": name,
                "seed": seed,
                "cost": cost,
                "mlflow_run_id": run_id,
                "validation": metrics,
            }
        )
        print(
            f"{name} seed={seed}: validation Recall@{config.k}="
            f"{metrics[f'recall@{config.k}']:.4f} NE={metrics['normalized_entropy']:.4f}",
            flush=True,
        )
        return metrics

    baseline = train_variant("baseline", seeds[0])
    decision = search(train, baseline, lambda name, w: train_variant(name, seeds[0], w), config.k)
    selected = decision["selected"]
    # Selection is written before the test partition is opened.
    frozen = {
        "selected": selected,
        "selection_seed": seeds[0],
        "audit": decision["audit"],
        "config": asdict(config),
        "seeds": list(seeds),
        "manifest": manifest,
        "gate": "validation recall >= baseline + 0.001 and NE <= baseline * 1.02",
    }
    (output / "decision.json").write_text(json.dumps(frozen, indent=2))
    for seed in seeds:
        if seed != seeds[0]:
            train_variant("baseline", seed)
            if selected != "baseline":
                train_variant(selected, seed, weights_for(train, selected))
        train_variant("no_content", seed, content=False)
    # Failed candidates are deliberately not scored on test.
    test = load_partition(data, "test")
    seen = np.concatenate([train, valid])
    for record in records:
        name, seed = record["variant"], record["seed"]
        if name in {"baseline", "no_content", selected}:
            record["test"] = assess(models[name, seed], test, seen, config.k)
            with mlflow.start_run(run_id=record["mlflow_run_id"]):
                mlflow.log_metrics(
                    {
                        "test/" + k.replace("@", "_at_"): v
                        for k, v in compact(record["test"]).items()
                    }
                )
    paired = []
    for seed in seeds:
        base = next(r["test"] for r in records if r["variant"] == "baseline" and r["seed"] == seed)
        candidate = next(
            r["test"] for r in records if r["variant"] == selected and r["seed"] == seed
        )
        paired.append(
            {
                "comparison": "selected_minus_baseline",
                "seed": seed,
                **paired_interval(base, candidate),
            }
        )
        ablated = next(
            r["test"] for r in records if r["variant"] == "no_content" and r["seed"] == seed
        )
        paired.append(
            {
                "comparison": "no_content_minus_baseline",
                "seed": seed,
                **paired_interval(base, ablated),
            }
        )
    result = {
        "config": asdict(config),
        "seeds": list(seeds),
        "selected": selected,
        "manifest": manifest,
        "runs": records,
        "paired_bootstrap": paired,
        "popularity": {
            "validation": popularity(train, valid, train, len(genres), config.k),
            "test": popularity(train, test, seen, len(genres), config.k),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "device": "cpu",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(
                b"".join(
                    p.name.encode() + p.read_bytes()
                    for p in sorted(Path(__file__).parent.glob("*.py"))
                )
            ).hexdigest(),
            "packages": {
                p: version(p)
                for p in ("torch", "numpy", "transformers", "mlflow", "optuna", "langgraph")
            },
        },
        "total_wall_seconds": time.perf_counter() - start,
    }
    torch.save(models[selected, seeds[0]].state_dict(), output / "selected-model.pt")
    # Public report contains aggregate metrics only, no per-user records.
    for record in result["runs"]:
        for split in ("validation", "test"):
            if split in record:
                record[split] = compact(record[split])
    result["popularity"] = {s: compact(m) for s, m in result["popularity"].items()}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    from reclab.report import render

    render(result, output)
    return result


def tune(data, output, trials=4):
    if not 1 <= trials <= 8:
        raise ValueError("Tuning budget is 1..8 trials")
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a fresh tuning output directory")
    output.mkdir(parents=True)
    train, valid = load_partition(data, "train"), load_partition(data, "valid")
    genres = load_genres(data)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        config = Config(
            epochs=4,
            dim=trial.suggest_categorical("dim", [16, 32]),
            lr=trial.suggest_float("lr", 0.001, 0.01, log=True),
        )
        model, cost = fit(train, genres, 42, config)
        metrics = assess(model, valid, train, config.k)
        trial.set_user_attr("cost", cost)
        trial.set_user_attr("normalized_entropy", metrics["normalized_entropy"])
        return metrics["recall@10"]

    study.optimize(objective, n_trials=trials)
    summary = {
        "best_params": study.best_params,
        "best_validation_recall": study.best_value,
        "trials": [
            {"number": t.number, "params": t.params, "value": t.value, "attributes": t.user_attrs}
            for t in study.trials
        ],
        "note": "Exploratory validation-only tuning; not applied to the fixed benchmark.",
    }
    (output / "tuning.json").write_text(json.dumps(summary, indent=2))
    return summary
