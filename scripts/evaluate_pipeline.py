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
  * Recall@k bar, MRR bar
  * Prompt-length comparison

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
        }
        results.append(res)
        recalls_by_game[target.game_id].append(recall_k)
        mrr_by_game[target.game_id].append(rr)

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

    # Baseline: nn + template
    print("Evaluating baseline: policy=nn, memory_format=template")
    agg_base, res_base = evaluate_config(
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
    with (run_dir / "results_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(res_base, f, ensure_ascii=False, indent=2)
    with (run_dir / "aggregate_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(agg_base, f, ensure_ascii=False, indent=2)

    # Current config
    print(f"Evaluating current: policy={args.policy}, memory_format={args.memory_format}")
    agg_cur, res_cur = evaluate_config(
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
    with (run_dir / "results_current.json").open("w", encoding="utf-8") as f:
        json.dump(res_cur, f, ensure_ascii=False, indent=2)
    with (run_dir / "aggregate_current.json").open("w", encoding="utf-8") as f:
        json.dump(agg_cur, f, ensure_ascii=False, indent=2)

    # Combined summary
    combined = {"baseline": agg_base, "current": agg_cur}
    with (run_dir / "aggregate_combined.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # Minimal visualizations: grouped bars + current prompt sizes
    viz_bar(
        run_dir,
        title=f"Recall@{args.k} (avg)",
        labels=["baseline", "current"],
        values=[agg_base["avg_recall_at_k"], agg_cur["avg_recall_at_k"]],
        filename="recall_bar.html",
        y_title=f"Recall@{args.k}",
    )
    viz_bar(
        run_dir,
        title="MRR (avg)",
        labels=["baseline", "current"],
        values=[agg_base["avg_mrr"], agg_cur["avg_mrr"]],
        filename="mrr_bar.html",
        y_title="MRR",
    )
    viz_prompt_lengths(run_dir, agg_cur["avg_prompt_sizes"])  # focuses on current config’s prompts

    print(f"Saved evaluation to: {run_dir}")
    print(
        f"Baseline: Recall@{args.k}={agg_base['avg_recall_at_k']:.3f}, MRR={agg_base['avg_mrr']:.3f} | "
        f"Current: Recall@{args.k}={agg_cur['avg_recall_at_k']:.3f}, MRR={agg_cur['avg_mrr']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
