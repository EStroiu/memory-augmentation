#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import plotly.graph_objects as go 
    from plotly.offline import plot as plotly_offline_plot
except Exception:
    go = None 
    plotly_offline_plot = None


@dataclass
class ClassifiedExample:
    true_role: str | None
    pred_role: str | None
    game_id: str | None
    quest: int | None
    memory_format: str | None
    prompt_stats: Dict[str, Any]
    llm_raw: Dict[str, Any]
    target: Dict[str, Any]


def load_results(run_dir: Path, which: str = "current") -> List[ClassifiedExample]:
    """Load `results_current.json` or `results_baseline.json` from a run subdir.

    Args:
        run_dir: Path to a single run directory, e.g. `.../runs/run_1`.
        which: "current" or "baseline".
    """
    fname = f"results_{which}.json"
    path = run_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    examples: List[ClassifiedExample] = []
    for r in raw:
        clf = r.get("classification", {}) or {}
        target = r.get("target", {}) or {}
        examples.append(
            ClassifiedExample(
                true_role=clf.get("true_role"),
                pred_role=clf.get("pred_role"),
                game_id=target.get("game_id"),
                quest=target.get("quest"),
                memory_format=r.get("memory_format"),
                prompt_stats=r.get("prompt_stats", {}),
                llm_raw=r.get("llm", {}),
                target=target,
            )
        )
    return examples


def build_confusion_counts(examples: List[ClassifiedExample]) -> Dict[Tuple[str, str], int]:
    """Return a flat confusion map (true_role, pred_role) -> count.

    Entries where either label is missing are skipped.
    """
    counts: Dict[Tuple[str, str], int] = {}
    for ex in examples:
        if not ex.true_role or not ex.pred_role:
            continue
        key = (str(ex.true_role), str(ex.pred_role))
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_top_confusions(counts: Dict[Tuple[str, str], int], top_n: int) -> None:
    """Print the most frequent (true_role, pred_role) mismatches."""
    items = [((tr, pr), c) for (tr, pr), c in counts.items() if tr != pr]
    items.sort(key=lambda x: x[1], reverse=True)
    if not items:
        print("No misclassifications found (all true_role == pred_role or labels missing).")
        return
    print(f"Top {min(top_n, len(items))} confusion pairs (true_role -> pred_role):")
    for (tr, pr), c in items[: top_n]:
        print(f"  {tr:>10} -> {pr:<10} : {c}")


def select_examples(
    examples: List[ClassifiedExample],
    true_role: str,
    pred_role: str,
    max_examples: int,
) -> List[ClassifiedExample]:
    out: List[ClassifiedExample] = []
    for ex in examples:
        if ex.true_role == true_role and ex.pred_role == pred_role:
            out.append(ex)
            if len(out) >= max_examples:
                break
    return out


def print_examples(examples: List[ClassifiedExample]) -> None:
    if not examples:
        print("No matching examples found.")
        return
    for i, ex in enumerate(examples, start=1):
        print("-" * 80)
        print(f"Example {i}")
        print(f"  game_id: {ex.game_id}  quest: {ex.quest}")
        print(f"  true_role: {ex.true_role}  pred_role: {ex.pred_role}")
        pstats = ex.prompt_stats or {}
        wm = pstats.get("with_memory", {}) or {}
        bl = pstats.get("baseline", {}) or {}
        print("  prompt_sizes.with_memory:", wm)
        print("  prompt_sizes.baseline:", bl)
        pred_text = (ex.llm_raw.get("prediction") or "").strip().replace("\n", " ")
        print("  llm_prediction (truncated):", repr(pred_text[:200]))


def viz_confusion_heatmap(counts: Dict[Tuple[str, str], int], out_html: Path) -> None:
    """Create a simple confusion heatmap using Plotly (if available)."""
    if go is None or plotly_offline_plot is None:
        print("Plotly not available; skipping heatmap.")
        return
    roles = sorted({tr for tr, _ in counts.keys()} | {pr for _, pr in counts.keys()})
    if not roles:
        print("No roles to visualize.")
        return
    n = len(roles)
    idx = {r: i for i, r in enumerate(roles)}
    import numpy as np  # local import guarded by outer try

    M = np.zeros((n, n), dtype=float)
    for (tr, pr), c in counts.items():
        if tr not in idx or pr not in idx:
            continue
        M[idx[tr], idx[pr]] += c
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, np.where(row_sums == 0, 1.0, row_sums), where=(row_sums != 0))
    fig = go.Figure(data=go.Heatmap(
        z=norm,
        x=roles,
        y=roles,
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Row-normalized"),
    ))
    fig.update_layout(title="LLM confusion heatmap (true vs predicted)", xaxis_title="Predicted role", yaxis_title="True role")
    plotly_offline_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")
    print(f"Saved confusion heatmap to: {out_html}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explainability analysis for LLM proposer-role predictions.")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to a single run directory (e.g., outputs/eval/.../runs/run_1)")
    ap.add_argument("--which", type=str, default="current", choices=["current", "baseline"], help="Analyze 'current' (with memory) or 'baseline' results.")
    ap.add_argument("--mode", type=str, default="top_confusions", choices=["top_confusions", "examples", "heatmap"], help="What to compute/visualize.")
    ap.add_argument("--top_n", type=int, default=10, help="Number of top confusion pairs to print (top_confusions mode).")
    ap.add_argument("--true_role", type=str, default=None, help="True role to filter on (examples mode).")
    ap.add_argument("--pred_role", type=str, default=None, help="Predicted role to filter on (examples mode).")
    ap.add_argument("--max_examples", type=int, default=5, help="Maximum number of examples to show (examples mode).")
    ap.add_argument("--heatmap_html", type=str, default="confusion_explain.html", help="Output HTML filename for heatmap (heatmap mode).")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        return 1

    try:
        examples = load_results(run_dir, which=str(args.which))
    except Exception as e:
        print(f"Failed to load results: {e}")
        return 1

    counts = build_confusion_counts(examples)

    if args.mode == "top_confusions":
        print_top_confusions(counts, int(args.top_n))
    elif args.mode == "examples":
        if not args.true_role or not args.pred_role:
            print("--true_role and --pred_role are required in examples mode.")
            return 1
        sel = select_examples(examples, str(args.true_role), str(args.pred_role), int(args.max_examples))
        print_examples(sel)
    elif args.mode == "heatmap":
        out_html = run_dir / str(args.heatmap_html)
        viz_confusion_heatmap(counts, out_html)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
