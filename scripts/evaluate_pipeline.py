#!/usr/bin/env python3
"""
Evaluate retrieval and prompting pipeline over an Avalon dataset.

Run this script from the repository root (after installing requirements).
Typical usage examples are in the top-level README.

Features:
- Runs retrieval probes (Recall@k, MRR) across all games in a dataset directory
- Supports two retrieval policies: nearest-neighbor (nn) and temporally-weighted (temporal)
- Supports two memory formats: template-only and template+heuristic-summary
- Optionally calls an LLM (OpenAI) for downstream role-prediction (stubbed by default)
- Computes prompt size stats (chars/words/lines) for baseline vs memory-augmented
- Saves per-run JSON results and Plotly HTML visualizations:
    * Per-role grouped bars: Recall@k by role, MRR by role
    * Confusion matrices (baseline/current) for proposer-role prediction proxy
    * Prompt-length comparison (with_memory vs baseline)

Notes:
- No per-quest plots are generated.
- Overall aggregate bars (Recall@k, MRR) are omitted in favor of per-role breakdowns.

Usage (from repo root):
        python scripts/evaluate_pipeline.py \
        --data_dir avalon-nlu/dataset \
        --k 3 \
        --policy nn \
        --memory_format template \
        --outdir outputs/eval

Optional:
    --policy temporal --alpha 0.5
    --memory_format template+summary
    --llm --openai-model gpt-4o-mini --openai-api-key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import numpy as np
except ImportError:
    print("numpy required. pip install -r requirements.txt", file=sys.stderr)
    raise

# ML deps (lazy handled)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

# Plotly optional
try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.offline import plot as plotly_offline_plot  # type: ignore
except Exception:
    go = None  # type: ignore
    plotly_offline_plot = None  # type: ignore


# Ensure repo root is importable when running as a script (python scripts/...) 
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.memory_utils import (
    MemoryEntry,
    load_game_json,
    build_memory_entries,
    build_embeddings,
    build_faiss_ip_index,
    retrieve_top,
    rerank_temporal,
    assemble_prompt,
    assemble_baseline_prompt,
    prompt_stats,
    pca_2d,
)


def llm_role_predict(prompt: str, use_llm: bool, model_name: str, api_key: str | None) -> Dict[str, Any]:
    """Optional LLM call for role prediction. Stubbed by default."""
    if not use_llm:
        return {"prediction": None, "cost": 0.0, "tokens": 0}
    try:
        import openai  # type: ignore
    except Exception:
        return {"prediction": None, "error": "openai library not installed", "cost": 0.0, "tokens": 0}
    if not api_key:
        return {"prediction": None, "error": "OPENAI_API_KEY not provided", "cost": 0.0, "tokens": 0}
    # For safety, we keep this as a no-op placeholder.
    # Implement your actual API call here if desired.
    return {"prediction": None, "note": "stubbed"}


# ---------- Visualization helpers ----------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def viz_bar(run_dir: Path, title: str, labels: List[str], values: List[float], filename: str, y_title: str) -> None:
    if go is None or plotly_offline_plot is None:
        return
    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(title=title, xaxis_title="Group", yaxis_title=y_title)
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")


def viz_boxplot(run_dir: Path, title: str, groups: List[str], values_per_group: List[List[float]], filename: str, y_title: str) -> None:
    if go is None or plotly_offline_plot is None:
        return
    fig = go.Figure()
    for grp, vals in zip(groups, values_per_group):
        fig.add_trace(go.Box(y=vals, name=grp))
    fig.update_layout(title=title, yaxis_title=y_title)
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")


def viz_prompt_lengths(run_dir: Path, avg_stats: Dict[str, Dict[str, float]]) -> None:
    if go is None or plotly_offline_plot is None:
        return
    metrics = ["chars", "words", "lines"]
    wm = [avg_stats["with_memory"][m] for m in metrics]
    bl = [avg_stats["baseline"][m] for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="with_memory", x=metrics, y=wm))
    fig.add_trace(go.Bar(name="baseline", x=metrics, y=bl))
    fig.update_layout(barmode="group", title="Average prompt sizes", yaxis_title="Count")
    plotly_offline_plot(fig, filename=str(run_dir / "prompt_lengths.html"), auto_open=False, include_plotlyjs="cdn")


def viz_embedding_pca_per_game(run_dir: Path, entries: List[MemoryEntry], emb: np.ndarray) -> None:
    if go is None or plotly_offline_plot is None:
        return
    # group indices by game
    by_game: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        by_game[e.game_id].append(i)
    pts = pca_2d(emb)
    for gid, idxs in by_game.items():
        x = pts[idxs, 0]
        y = pts[idxs, 1]
        texts = [f"{entries[i].entry_id}" for i in idxs]
        colors = [entries[i].quest for i in idxs]
        fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="markers", marker=dict(color=colors, colorscale="Viridis", size=9), text=texts, hoverinfo="text")])
        fig.update_layout(title=f"Embeddings PCA — {gid}")
        plotly_offline_plot(fig, filename=str(run_dir / f"pca_{gid}.html"), auto_open=False, include_plotlyjs="cdn")


def viz_confusion_matrix(
    run_dir: Path,
    title: str,
    roles: List[str],
    confusion: Dict[str, Dict[str, float]],
    f1_by_role: Dict[str, float] | None,
    filename: str,
) -> None:
    if go is None or plotly_offline_plot is None:
        return
    n = len(roles)
    # Build count matrix
    M = np.zeros((n, n), dtype=float)
    for i, tr in enumerate(roles):
        row = confusion.get(tr, {}) or {}
        for j, pr in enumerate(roles):
            M[i, j] = float(row.get(pr, 0.0))
    # Normalize rows to probabilities (avoid division by zero)
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, np.where(row_sums == 0, 1.0, row_sums), where=np.ones_like(M, dtype=bool))
    # Text annotations: show percentage and (count). On diagonal, also show F1 if provided
    text = []
    for i in range(n):
        row = []
        for j in range(n):
            pct = norm[i, j] * 100.0
            cell = f"{pct:.1f}%\n({int(M[i, j])})"
            if i == j and f1_by_role is not None:
                role = roles[i]
                f1 = float(f1_by_role.get(role, 0.0)) if role in f1_by_role else 0.0
                cell = f"{pct:.1f}%\n({int(M[i, j])})\nF1={f1:.2f}"
            row.append(cell)
        text.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=norm,
        x=roles,
        y=roles,
        colorscale="Blues",
        reversescale=False,
        zmin=0.0,
        zmax=1.0,
        text=text,
        texttemplate="%{text}",
        hoverinfo="skip",
        showscale=True,
        colorbar=dict(title="Row-normalized")
    ))
    fig.update_layout(title=title, xaxis_title="Predicted role", yaxis_title="True role")
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")


def viz_grouped_bars(
    run_dir: Path,
    title: str,
    categories: List[str],
    baseline_values: List[float],
    current_values: List[float],
    filename: str,
    y_title: str,
) -> None:
    if go is None or plotly_offline_plot is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(name="baseline", x=categories, y=baseline_values))
    fig.add_trace(go.Bar(name="current", x=categories, y=current_values))
    fig.update_layout(barmode="group", title=title, xaxis_title="Category", yaxis_title=y_title)
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")


def avg_numbers(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def merge_avg_prompt_sizes(items: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    out = {"with_memory": {"chars": 0.0, "words": 0.0, "lines": 0.0},
           "baseline": {"chars": 0.0, "words": 0.0, "lines": 0.0}}
    if not items:
        return out
    for key1 in out.keys():
        for key2 in out[key1].keys():
            out[key1][key2] = float(np.mean([x.get(key1, {}).get(key2, 0.0) for x in items]))
    return out


def average_aggregates(aggs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not aggs:
        return {}
    # Simple numeric fields
    avg_recall = avg_numbers([a.get("avg_recall_at_k", 0.0) for a in aggs])
    avg_mrr = avg_numbers([a.get("avg_mrr", 0.0) for a in aggs])
    avg_prompt_sizes = merge_avg_prompt_sizes([a.get("avg_prompt_sizes", {}) for a in aggs])
    # Merge by_role and by_quest with mean
    roles = set()
    quests = set()
    for a in aggs:
        roles |= set(a.get("by_role", {}).keys())
        quests |= set(a.get("by_quest", {}).keys())
    by_role = {}
    for r in sorted(roles):
        vals_r = [a.get("by_role", {}).get(r, {}) for a in aggs]
        by_role[r] = {
            "count": int(np.mean([v.get("count", 0) for v in vals_r])) if vals_r else 0,
            "recall_at_k": avg_numbers([v.get("recall_at_k", 0.0) for v in vals_r]),
            "mrr": avg_numbers([v.get("mrr", 0.0) for v in vals_r]),
        }
    by_quest = {}
    for q in sorted(quests):
        vals_q = [a.get("by_quest", {}).get(q, {}) for a in aggs]
        by_quest[q] = {
            "count": int(np.mean([v.get("count", 0) for v in vals_q])) if vals_q else 0,
            "recall_at_k": avg_numbers([v.get("recall_at_k", 0.0) for v in vals_q]),
            "mrr": avg_numbers([v.get("mrr", 0.0) for v in vals_q]),
        }
    # Classification averages
    roles_clf = set()
    for a in aggs:
        roles_clf |= set(a.get("clf_by_role", {}).keys())
    clf_by_role = {}
    for r in sorted(roles_clf):
        vals = [a.get("clf_by_role", {}).get(r, {}) for a in aggs]
        clf_by_role[r] = {
            "precision": avg_numbers([v.get("precision", 0.0) for v in vals]),
            "recall": avg_numbers([v.get("recall", 0.0) for v in vals]),
            "f1": avg_numbers([v.get("f1", 0.0) for v in vals]),
        }
    micro_f1 = avg_numbers([a.get("clf_micro_f1", 0.0) for a in aggs])

    # Sum confusion matrices across runs (later normalize for plotting)
    conf_roles = sorted(roles_clf)
    conf: Dict[str, Dict[str, float]] = {tr: {pr: 0.0 for pr in conf_roles} for tr in conf_roles}
    for a in aggs:
        cm = a.get("clf_confusion", {}) or {}
        for tr in conf_roles:
            row = cm.get(tr, {}) or {}
            for pr in conf_roles:
                conf[tr][pr] += float(row.get(pr, 0))
    # Model, policy, etc. just echo from first
    base = aggs[0]
    return {
        "k": base.get("k"),
        "policy": base.get("policy"),
        "alpha": base.get("alpha"),
        "memory_format": base.get("memory_format"),
        "model": base.get("model"),
        "avg_recall_at_k": avg_recall,
        "avg_mrr": avg_mrr,
        "avg_prompt_sizes": avg_prompt_sizes,
        "by_role": by_role,
        "by_quest": by_quest,
        "clf_by_role": clf_by_role,
        "clf_micro_f1": micro_f1,
        "clf_confusion": conf,
    }


def evaluate_config(
    files: List[Path],
    k: int,
    policy: str,
    alpha: float,
    memory_format: str,
    model_name: str,
    use_llm: bool,
    openai_model: str,
    openai_api_key: str | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one configuration and return (aggregate, results).

    Retrieval metrics exclude the query (self) from candidate results to avoid trivial rank=1.
    Positives are defined as entries from the same game as the query (excluding self).
    """
    # Build entries across all games
    all_entries: List[MemoryEntry] = []
    per_game_indices: Dict[str, List[int]] = defaultdict(list)
    for f in files:
        game = load_game_json(f)
        gid = f.stem
        entries = build_memory_entries(game, gid, memory_format=memory_format)
        start_idx = len(all_entries)
        all_entries.extend(entries)
        per_game_indices[gid].extend(list(range(start_idx, start_idx + len(entries))))

    # Embeddings and index
    emb, model = build_embeddings(all_entries, model_name)
    index = build_faiss_ip_index(emb)

    results: List[Dict[str, Any]] = []
    recalls_by_game: Dict[str, List[float]] = defaultdict(list)
    mrr_by_game: Dict[str, List[float]] = defaultdict(list)
    # Also track by proposer role and by quest number
    recalls_by_role: Dict[str, List[float]] = defaultdict(list)
    mrr_by_role: Dict[str, List[float]] = defaultdict(list)
    recalls_by_quest: Dict[int, List[float]] = defaultdict(list)
    mrr_by_quest: Dict[int, List[float]] = defaultdict(list)
    # Classification counters for proposer-role prediction via retrieval (top-1 label transfer)
    roles_seen: set[str] = set()
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    # Confusion matrix counts: true_role -> pred_role -> count
    conf_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for qi, target in enumerate(all_entries):
        q_vec = model.encode([target.text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        # Retrieve more than k for stability
        D, I = retrieve_top(index, q_vec, topn=min(max(50, k * 5), len(all_entries)))

        # Exclude self from candidate pool
        mask = I != qi
        I = I[mask]
        D = D[mask]

        # Re-rank if temporal policy
        if policy == "temporal":
            reranked = rerank_temporal(all_entries, I, D, target, alpha=float(alpha))
            I2 = np.array([j for j, s in reranked], dtype=np.int64)
            D2 = np.array([s for j, s in reranked], dtype=np.float32)
        else:
            I2, D2 = I, D

        # Top-k selection
        topk_idx = list(I2[: k])
        topk_scores = list(D2[: k])

        # Metrics: define positives as same-game entries (excluding self)
        same_game_indices = {i for i, e in enumerate(all_entries) if (e.game_id == target.game_id and i != qi)}
        recall_k = 1.0 if any(j in same_game_indices for j in topk_idx) else 0.0
        # MRR: rank of closest same-game item
        rr = 0.0
        pos = next((p for p, j in enumerate(I2) if int(j) in same_game_indices), None)
        if pos is not None:
            rr = 1.0 / float(pos + 1)

        # Prompts and stats
        retrieved_pairs = [(all_entries[j], float(s)) for j, s in zip(topk_idx, topk_scores)]
        prompt_mem = assemble_prompt(target, retrieved_pairs)
        prompt_base = assemble_baseline_prompt(target)
        pstats = prompt_stats(prompt_mem, prompt_base)

        # Optional LLM call (stubbed)
        llm_out = llm_role_predict(prompt_mem, bool(use_llm), openai_model, openai_api_key)

        # Heuristic proposer-role prediction: transfer top-1 retrieved proposer_role
        true_role = target.proposer_role
        pred_role = None
        if len(I2) > 0:
            cand_role = all_entries[int(I2[0])].proposer_role
            if cand_role is not None:
                pred_role = cand_role
        if true_role is not None:
            roles_seen.add(true_role)
        if pred_role is not None:
            roles_seen.add(pred_role)
        # Update confusion counts when both labels known
        if true_role is not None and pred_role is not None:
            if pred_role == true_role:
                tp[true_role] += 1
            else:
                fp[pred_role] += 1
                fn[true_role] += 1
            conf_counts[true_role][pred_role] += 1

        res = {
            "target": asdict(target),
            "topk": [
                {"entry_id": all_entries[j].entry_id, "game_id": all_entries[j].game_id, "quest": int(all_entries[j].quest), "score": float(s)}
                for j, s in zip(topk_idx, topk_scores)
            ],
            "recall_at_k": recall_k,
            "rr": rr,
            "policy": policy,
            "alpha": float(alpha) if policy == "temporal" else None,
            "memory_format": memory_format,
            "prompt_stats": pstats,
            "llm": llm_out,
            "classification": {
                "true_role": true_role,
                "pred_role_top1": pred_role,
            },
        }
        results.append(res)
        recalls_by_game[target.game_id].append(recall_k)
        mrr_by_game[target.game_id].append(rr)
        role_key = (target.proposer_role or "unknown")
        recalls_by_role[role_key].append(recall_k)
        mrr_by_role[role_key].append(rr)
        recalls_by_quest[int(target.quest)].append(recall_k)
        mrr_by_quest[int(target.quest)].append(rr)

    # Aggregations
    avg_recall = float(np.mean([r["recall_at_k"] for r in results])) if results else 0.0
    avg_mrr = float(np.mean([r["rr"] for r in results])) if results else 0.0
    avg_prompt_sizes = {
        "with_memory": {
            "chars": float(np.mean([r["prompt_stats"]["with_memory"]["chars"] for r in results])) if results else 0.0,
            "words": float(np.mean([r["prompt_stats"]["with_memory"]["words"] for r in results])) if results else 0.0,
            "lines": float(np.mean([r["prompt_stats"]["with_memory"]["lines"] for r in results])) if results else 0.0,
        },
        "baseline": {
            "chars": float(np.mean([r["prompt_stats"]["baseline"]["chars"] for r in results])) if results else 0.0,
            "words": float(np.mean([r["prompt_stats"]["baseline"]["words"] for r in results])) if results else 0.0,
            "lines": float(np.mean([r["prompt_stats"]["baseline"]["lines"] for r in results])) if results else 0.0,
        },
    }

    # Summaries per category
    by_role_summary = {
        role: {
            "count": int(len(vals)),
            "recall_at_k": float(np.mean(vals)) if vals else 0.0,
            "mrr": float(np.mean(mrr_by_role.get(role, []))) if mrr_by_role.get(role, []) else 0.0,
        }
        for role, vals in recalls_by_role.items()
    }
    by_quest_summary = {
        int(q): {
            "count": int(len(vals)),
            "recall_at_k": float(np.mean(vals)) if vals else 0.0,
            "mrr": float(np.mean(mrr_by_quest.get(int(q), []))) if mrr_by_quest.get(int(q), []) else 0.0,
        }
        for q, vals in recalls_by_quest.items()
    }

    # Classification metrics (per-role F1 and micro-F1)
    roles_sorted = sorted(roles_seen)
    clf_by_role: Dict[str, Dict[str, float]] = {}
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for r in roles_sorted:
        t = float(tp.get(r, 0))
        f_p = float(fp.get(r, 0))
        f_n = float(fn.get(r, 0))
        tp_sum += int(t)
        fp_sum += int(f_p)
        fn_sum += int(f_n)
        prec = (t / (t + f_p)) if (t + f_p) > 0 else 0.0
        rec = (t / (t + f_n)) if (t + f_n) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        clf_by_role[r] = {"precision": prec, "recall": rec, "f1": f1, "tp": t, "fp": f_p, "fn": f_n}
    micro_prec = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_rec = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    agg = {
        "k": int(k),
        "policy": policy,
        "alpha": float(alpha) if policy == "temporal" else None,
        "memory_format": memory_format,
        "model": model_name,
        "avg_recall_at_k": avg_recall,
        "avg_mrr": avg_mrr,
        "avg_prompt_sizes": avg_prompt_sizes,
        "games": {gid: {"recall_at_k": float(np.mean(recalls_by_game[gid])) if recalls_by_game[gid] else 0.0,
                          "mrr": float(np.mean(mrr_by_game[gid])) if mrr_by_game[gid] else 0.0}
                   for gid in recalls_by_game.keys()},
        "by_role": by_role_summary,
        "by_quest": by_quest_summary,
        "clf_by_role": clf_by_role,
        "clf_micro_f1": micro_f1,
        "clf_confusion": {tr: dict(prs) for tr, prs in conf_counts.items()},
    }
    return agg, results


# ---------- Combined evaluation (baseline vs current) ----------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval and prompting pipeline over an Avalon dataset.")
    ap.add_argument("--data_dir", type=str, default="dataset", help="Directory containing *.json games")
    ap.add_argument("--k", type=int, default=3, help="Top-k for retrieval and Recall@k")
    ap.add_argument("--policy", type=str, default="nn", choices=["nn", "temporal"], help="Retrieval policy")
    ap.add_argument("--alpha", type=float, default=0.5, help="Temporal decay alpha for 'temporal' policy")
    ap.add_argument("--memory_format", type=str, default="template", choices=["template", "template+summary"], help="Memory format for entries")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    ap.add_argument("--outdir", type=str, default="outputs/eval", help="Base output directory for saving results and visuals")
    ap.add_argument("--llm", action="store_true", help="Call OpenAI LLM for downstream role prediction (stubbed by default)")
    ap.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model name (if --llm)")
    ap.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key (if --llm)")
    ap.add_argument("--max_games", type=int, default=None, help="Limit number of games for quick tests")
    ap.add_argument("--num_runs", type=int, default=1, help="Repeat the experiment N times and average results")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed (incremented per run)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        return 1

    files = sorted(data_dir.glob("*.json"))
    if args.max_games:
        files = files[: int(args.max_games)]
    if not files:
        print(f"No JSON games in {data_dir}", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / f"{ts}_k-{args.k}_policy-{args.policy}_mem-{args.memory_format}"
    ensure_dir(run_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: getattr(args, k) for k in vars(args)}, f, indent=2)

    # Multi-run loop: save per-run artifacts in subfolders and average aggregates
    agg_base_runs: List[Dict[str, Any]] = []
    agg_cur_runs: List[Dict[str, Any]] = []
    for i in range(int(args.num_runs)):
        print(f"Run {i+1}/{args.num_runs}")
        run_subdir = run_dir / f"runs/run_{i+1}"
        ensure_dir(run_subdir)
        # Optional: set seeds (might not affect deterministic parts but harmless)
        try:
            np.random.seed(int(args.seed) + i)
        except Exception:
            pass

        # Baseline
        print("  Evaluating baseline: policy=nn, memory_format=template")
        agg_base_i, res_base_i = evaluate_config(
            files=files,
            k=int(args.k),
            policy="nn",
            alpha=0.0,
            memory_format="template",
            model_name=args.model,
            use_llm=bool(args.llm),
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
        )
        agg_base_runs.append(agg_base_i)
        with (run_subdir / "results_baseline.json").open("w", encoding="utf-8") as f:
            json.dump(res_base_i, f, ensure_ascii=False, indent=2)
        with (run_subdir / "aggregate_baseline.json").open("w", encoding="utf-8") as f:
            json.dump(agg_base_i, f, ensure_ascii=False, indent=2)

        # Current
        print(f"  Evaluating current: policy={args.policy}, memory_format={args.memory_format}")
        agg_cur_i, res_cur_i = evaluate_config(
            files=files,
            k=int(args.k),
            policy=args.policy,
            alpha=float(args.alpha),
            memory_format=args.memory_format,
            model_name=args.model,
            use_llm=bool(args.llm),
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
        )
        agg_cur_runs.append(agg_cur_i)
        with (run_subdir / "results_current.json").open("w", encoding="utf-8") as f:
            json.dump(res_cur_i, f, ensure_ascii=False, indent=2)
        with (run_subdir / "aggregate_current.json").open("w", encoding="utf-8") as f:
            json.dump(agg_cur_i, f, ensure_ascii=False, indent=2)

    # Averages across runs
    agg_base = average_aggregates(agg_base_runs)
    agg_cur = average_aggregates(agg_cur_runs)
    with (run_dir / "aggregate_baseline_mean.json").open("w", encoding="utf-8") as f:
        json.dump(agg_base, f, ensure_ascii=False, indent=2)
    with (run_dir / "aggregate_current_mean.json").open("w", encoding="utf-8") as f:
        json.dump(agg_cur, f, ensure_ascii=False, indent=2)
    combined = {"baseline": agg_base, "current": agg_cur}
    with (run_dir / "aggregate_combined_mean.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # Write F1 results as CSV (and keep txt for convenience)
    roles = sorted(set(list(agg_base.get("clf_by_role", {}).keys()) + list(agg_cur.get("clf_by_role", {}).keys())))
    if roles:
        # CSV
        import csv
        with (run_dir / "role_f1_table.csv").open("w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["approach", *roles, "micro_f1"])
            writer.writerow(["baseline", *[f"{agg_base['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles], f"{agg_base.get('clf_micro_f1', 0.0):.3f}"])
            writer.writerow(["current", *[f"{agg_cur['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles], f"{agg_cur.get('clf_micro_f1', 0.0):.3f}"])
        # TXT (tab-delimited)
        header = ["approach"] + roles + ["micro_f1"]
        lines = ["\t".join(header)]
        base_vals = ["baseline"] + [f"{agg_base['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles] + [f"{agg_base.get('clf_micro_f1', 0.0):.3f}"]
        cur_vals = ["current"] + [f"{agg_cur['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles] + [f"{agg_cur.get('clf_micro_f1', 0.0):.3f}"]
        (run_dir / "role_f1_table.txt").write_text("\n".join([*lines, "\t".join(base_vals), "\t".join(cur_vals)]) + "\n", encoding="utf-8")

        # Optional: save per-run F1s for transparency (if num_runs > 1)
        if int(getattr(args, "num_runs", 1)) > 1:
            with (run_dir / "role_f1_table_runs.csv").open("w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["run", "approach", "role", "f1"])
                # baseline
                for i, a in enumerate(agg_base_runs, start=1):
                    for r in roles:
                        writer.writerow([i, "baseline", r, f"{a.get('clf_by_role', {}).get(r, {}).get('f1', 0.0):.3f}"])
                # current
                for i, a in enumerate(agg_cur_runs, start=1):
                    for r in roles:
                        writer.writerow([i, "current", r, f"{a.get('clf_by_role', {}).get(r, {}).get('f1', 0.0):.3f}"])

    # Visualizations: per-role grouped bars, confusion matrices, and current prompt sizes
    viz_prompt_lengths(run_dir, agg_cur["avg_prompt_sizes"])  # focuses on current config’s prompts

    # New: Per-role comparison (by proposer role)
    roles = sorted(set(list(agg_base.get("by_role", {}).keys()) + list(agg_cur.get("by_role", {}).keys())))
    if roles:
        base_recall = [agg_base["by_role"].get(r, {}).get("recall_at_k", 0.0) for r in roles]
        cur_recall = [agg_cur["by_role"].get(r, {}).get("recall_at_k", 0.0) for r in roles]
        viz_grouped_bars(
            run_dir,
            title=f"Recall@{args.k} by proposer role",
            categories=roles,
            baseline_values=base_recall,
            current_values=cur_recall,
            filename="recall_by_role.html",
            y_title=f"Recall@{args.k}",
        )
        base_mrr = [agg_base["by_role"].get(r, {}).get("mrr", 0.0) for r in roles]
        cur_mrr = [agg_cur["by_role"].get(r, {}).get("mrr", 0.0) for r in roles]
        viz_grouped_bars(
            run_dir,
            title="MRR by proposer role",
            categories=roles,
            baseline_values=base_mrr,
            current_values=cur_mrr,
            filename="mrr_by_role.html",
            y_title="MRR",
        )

    # Note: Per-quest plots intentionally omitted per requirements.

    # New: Confusion matrices by role (averaged across runs)
    cm_roles = sorted(set(list(agg_base.get("clf_by_role", {}).keys()) + list(agg_cur.get("clf_by_role", {}).keys())))
    if cm_roles:
        # Baseline
        viz_confusion_matrix(
            run_dir,
            title="Confusion matrix (baseline)",
            roles=cm_roles,
            confusion=agg_base.get("clf_confusion", {}),
            f1_by_role={r: agg_base.get("clf_by_role", {}).get(r, {}).get("f1", 0.0) for r in cm_roles},
            filename="confusion_baseline.html",
        )
        # Current
        viz_confusion_matrix(
            run_dir,
            title="Confusion matrix (current)",
            roles=cm_roles,
            confusion=agg_cur.get("clf_confusion", {}),
            f1_by_role={r: agg_cur.get("clf_by_role", {}).get(r, {}).get("f1", 0.0) for r in cm_roles},
            filename="confusion_current.html",
        )

    print(f"Saved evaluation to: {run_dir}")
    print(
        f"Baseline (mean of {args.num_runs} runs): Recall@{args.k}={agg_base['avg_recall_at_k']:.3f}, MRR={agg_base['avg_mrr']:.3f} | "
        f"Current (mean of {args.num_runs} runs): Recall@{args.k}={agg_cur['avg_recall_at_k']:.3f}, MRR={agg_cur['avg_mrr']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
