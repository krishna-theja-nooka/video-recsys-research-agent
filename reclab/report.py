"""Portable reports generated exclusively from measured JSON."""

import html
from pathlib import Path

import numpy as np


def render(result, output):
    output = Path(output)
    k = result["config"]["k"]
    names = list(dict.fromkeys(r["variant"] for r in result["runs"] if "test" in r))
    summary = []
    for name in names:
        runs = [r for r in result["runs"] if r["variant"] == name and "test" in r]
        recall = [r["test"][f"recall@{k}"] for r in runs]
        ne = [r["test"]["normalized_entropy"] for r in runs]
        cost = [r["cost"]["train_wall_seconds"] for r in runs]
        summary.append(
            (
                name,
                np.mean(recall),
                np.std(recall, ddof=1) if len(runs) > 1 else 0,
                np.mean(ne),
                np.mean(cost),
                len(runs),
            )
        )
    pop = result["popularity"]["test"]
    text = [
        "# Research report: bounded data interventions in movie recommendation",
        "",
        "## Abstract",
        "",
        "This experiment tests a two-tower recommender with a small genre Transformer "
        "and a deterministic LangGraph data-intervention controller. MovieLens 100K is "
        "a movie-rating proxy for video discovery, not a short-video impression benchmark. "
        "All figures below are measured locally; no online engagement or production impact "
        "is claimed.",
        "",
        "## Protocol",
        "",
        "Interactions are globally sorted by timestamp, user and item. Timestamp boundaries "
        "near 80% and 90% define train/validation/test. Ties remain in one partition. "
        "Mappings and eligible catalog are fitted on train only. Unseen users/items in later "
        "partitions are excluded and counted. Ratings >=4 are positive; ratings 1–3 are "
        "observed negative labels, not unobserved impressions. Validation selects only the "
        "data intervention. Test is opened after decision.json is written; failed proposals "
        "are never scored on test. All hyperparameters and epochs remain fixed.",
        "",
        f"Selected intervention: **{result['selected']}**. Seeds: {result['seeds']}. "
        f"Epochs per model: {result['config']['epochs']}.",
        "",
        "## Measured held-out results",
        "",
        f"| Variant | Recall@{k} mean ± seed SD | NE mean ↓ | Train wall seconds mean | Seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, recall, sd, ne, seconds, seeds in summary:
        text.append(f"| {name} | {recall:.4f} ± {sd:.4f} | {ne:.4f} | {seconds:.2f} | {seeds} |")
    text += [
        f"| popularity | {pop[f'recall@{k}']:.4f} | {pop['normalized_entropy']:.4f} | "
        "no gradient training | deterministic |",
        "",
        "Popularity ranks training positive counts; its pointwise prediction uses item "
        "positive rates smoothed by 20 global-prior pseudo-observations. The no_content "
        "ablation removes the genre encoder, leaving ID embeddings and biases. This also "
        "changes parameter count and compute, so it is not a capacity-matched control.",
        "",
        "## Validation-only intervention trials",
        "",
        f"| Intervention | Recall@{k} | NE | Train seconds |",
        "|---|---:|---:|---:|",
    ]
    for r in result["runs"]:
        if r["seed"] == result["seeds"][0]:
            text.append(
                f"| {r['variant']} | {r['validation'][f'recall@{k}']:.4f} | "
                f"{r['validation']['normalized_entropy']:.4f} | "
                f"{r['cost']['train_wall_seconds']:.2f} |"
            )
    text += [
        "",
        "The controller tests exactly three allowlisted interventions: linear recency "
        "weights 0.8–1.2; weight 1.2 for items at or below median training item frequency; "
        "and weight 0.8 for rating-3 examples. No rows, labels, validation data or test "
        "data are edited. Approval requires validation Recall@K >= frozen baseline + "
        "0.001 and normalized entropy <= baseline × 1.02. The best qualifying Recall@K "
        "wins, otherwise the baseline is retained. The intervention is chosen with the "
        "first seed only, then replicated with remaining paired seeds.",
        "",
        "## Paired test uncertainty",
        "",
        "These are percentile 95% paired user-bootstrap intervals, conditional on each "
        "trained model and split (2,000 resamples). If selected equals baseline, its zero interval is structural, not evidence of statistical equivalence. They are not across-dataset or causal "
        "confidence intervals. Seed variability is reported separately above.",
        "",
        "| Comparison | Seed | Recall delta | Lower | Upper |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in result["paired_bootstrap"]:
        text.append(
            f"| {p['comparison']} | {p['seed']} | {p['mean_delta']:.4f} | {p['lower']:.4f} | {p['upper']:.4f} |"
        )
    total_train = sum(r["cost"]["train_wall_seconds"] for r in result["runs"])
    text += [
        "",
        "## Training-cost accounting",
        "",
        f"Total measured gradient-training wall time: {total_train:.2f} seconds across "
        f"{len(result['runs'])} training runs, including rejected proposals. Full experiment "
        f"wall time: {result['total_wall_seconds']:.2f} seconds including validation, tracking "
        "and final scoring (excluding report rendering and checkpoint serialization). "
        "results.json records optimizer steps, examples processed, parameter counts and CPU "
        "process time per run. CPU time can exceed wall time with multiple threads. There "
        "is no measured energy, GPU-hour, cloud-dollar or peak-memory claim. Optional "
        "Optuna tuning is separate and additional, with costs in tuning.json.",
        "",
        "## Metrics",
        "",
        f"Recall@{k} is macro-averaged over eligible users with at least one held-out "
        "positive: hits / all their held-out positives. Ranking uses the full training "
        "catalog, excluding earlier observed items (train for validation; train+validation "
        "for test). Lower ratings in the evaluation partition remain ranking candidates. "
        "NDCG uses binary relevance. Stable tie-breaking uses mapped item IDs.",
        "",
        "Normalized entropy = mean binary cross-entropy / H(p), where p is the positive "
        "fraction of the evaluation partition. It is undefined for single-class labels; "
        "the code raises an error. NE=1 is the evaluation-prevalence constant predictor. "
        "This is rating-label NE, not CTR normalized entropy on a representative impression "
        "sample. No negative sampling is used; calibration cannot be generalized to clicks.",
        "",
        "## Data coverage",
        "",
        "| Partition | Original | Warm-start retained | Cold-start excluded |",
        "|---|---:|---:|---:|",
    ]
    for name, p in result["manifest"]["partitions"].items():
        text.append(f"| {name} | {p['raw']} | {p['retained']} | {p['excluded_cold_start']} |")
    text += [
        "",
        "## Limitations and interpretation",
        "",
        "This small, historical, self-selected ratings dataset is not a randomized "
        "exposure log. Global temporal splitting excludes cold-start users and items, "
        "so accuracy describes only the retained warm-start population. Genre tokens are "
        "metadata, not video pixels or natural-language understanding. The Transformer "
        "is initialized randomly and trained from scratch; no pretrained model is "
        "downloaded. This is a deterministic experimental agent, not an LLM agent or "
        "autonomous deployment system. Multiple validation trials introduce selection "
        "bias. Three seeds are a small sample. A qualifying validation change may fail "
        "to improve test performance. Negative findings are retained. No result establishes "
        "Meta-scale performance, PhD-level novelty, or causality.",
        "",
        "The benchmark keeps the baseline fixed. Optuna is an optional exploratory study "
        "that never accesses test or automatically changes the benchmark. A stronger "
        "study would preregister a new baseline, add capacity-matched encoders and multiple "
        "datasets, then validate on licensed video exposure/watch-time data.",
        "",
        "## Reproduction",
        "",
        "See README.md for pinned installation and commands. Raw data and checkpoints "
        "are excluded from distribution. Results include environment versions and split "
        "digests. CPU timings depend on hardware and load; equal seeds do not guarantee "
        "bitwise equality across different platforms.",
        "",
        "## References",
        "",
        "- Harper, F. M. and Konstan, J. A. (2015). The MovieLens Datasets: History and "
        "Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872",
        "- https://grouplens.org/datasets/movielens/100k/",
        "- https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        "- https://huggingface.co/docs/transformers/model_doc/bert",
        "- https://docs.langchain.com/oss/python/langgraph/overview",
        "",
    ]
    (output / "research-report.md").write_text("\n".join(text), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{r:.4f} ± {sd:.4f}</td>"
        f"<td>{ne:.4f}</td><td>{t:.1f}s</td></tr>"
        for n, r, sd, ne, t, _ in summary
    )
    bars = "".join(
        f'<div class="barrow"><span>{html.escape(n)}</span><div class="track">'
        f'<div class="bar" style="width:{r * 100:.2f}%"></div></div>'
        f"<strong>{r:.3f}</strong></div>"
        for n, r, *_ in summary
    )
    doc = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FrameLab · Research</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f1e9;color:#182c30;font:16px/1.6 system-ui}
main{max-width:1100px;margin:auto;padding:55px 24px}header{border-bottom:1px solid #bccac5;padding-bottom:30px}
small{letter-spacing:.16em;text-transform:uppercase;color:#35716a}h1{font-size:clamp(32px,6vw,60px);line-height:1.1;margin:18px 0}
h2{font-size:23px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:28px 0}
.card,section{background:#fff;border:1px solid #d3ddd8;border-radius:14px;padding:24px;margin-bottom:20px}
.card b{display:block;font-size:30px}.badge{background:#dceae2;padding:8px 16px;border-radius:30px;display:inline-block}
table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:12px;border-bottom:1px solid #dde5df}
.scroll{overflow:auto}.barrow{display:grid;grid-template-columns:130px 1fr 55px;gap:16px;align-items:center;margin:20px 0}
.track{background:#eef1ed;height:20px;border-radius:4px}.bar{height:100%;background:#23776a;border-radius:4px}
footer{color:#506563}a{color:#16685d}@media(max-width:650px){.cards{grid-template-columns:1fr}.barrow{grid-template-columns:90px 1fr 45px}}</style>
<main><header><small>Project 09 / Recommendation research</small><h1>FrameLab</h1>
<p>Deep learning recommendations. Bounded experiments. Measured evidence.</p>
<span class="badge">MovieLens proxy · CPU benchmark · No paid services</span></header>"""
    doc += f"""<div class="cards"><div class="card"><small>Selected intervention</small><b>{html.escape(result["selected"])}</b>Validation gate decision</div>
<div class="card"><small>Replication</small><b>{len(result["seeds"])} seeds</b>Paired initialization and shuffling</div>
<div class="card"><small>Experiment wall time</small><b>{result["total_wall_seconds"] / 60:.1f} min</b>Measured on this machine</div></div>
<section><h2>Held-out results</h2><div class="scroll"><table><tr><th>Model</th><th>Recall@{k} ± seed SD</th><th>NE ↓</th><th>Train wall mean</th></tr>{rows}</table></div>
<p>Popularity Recall@{k}: {pop[f"recall@{k}"]:.4f} · NE: {pop["normalized_entropy"]:.4f}.</p></section>
<section><h2>Recall@{k}</h2><p>Full catalog ranking · axis 0–1 · higher is better</p>{bars}</section>
<section><h2>What this demonstrates</h2><p>A fixed temporal split, an auditable three-trial LangGraph controller, train-only reweighting, and honest held-out evaluation.</p>
<p>Movie ratings are a proxy for discovery. These are not short-video watch-time, CTR, or online engagement results. Selection on validation does not guarantee improvement on test.</p>
<a href="research-report.md">Read the full research report</a> · <a href="results.json">Inspect measured results</a></section>
<footer>Generated from local experiment outputs. No raw user interactions or model weights are embedded.</footer></main></html>"""
    (output / "index.html").write_text(doc, encoding="utf-8")
