#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go 
from plotly.offline import plot as plotly_offline_plot

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


def compute_role_metrics(examples: List[ClassifiedExample]) -> Dict[str, Dict[str, float]]:
    """Compute per-role precision, recall, and F1 from classified examples.

    This re-derives metrics directly from the per-example outputs of a single run,
    which is useful for sanity-checking aggregate JSON and for slicing by run.
    """
    roles = sorted({ex.true_role for ex in examples if ex.true_role} | {ex.pred_role for ex in examples if ex.pred_role})
    if not roles:
        return {}
    tp = {r: 0 for r in roles}
    fp = {r: 0 for r in roles}
    fn = {r: 0 for r in roles}
    for ex in examples:
        if not ex.true_role or not ex.pred_role:
            continue
        t, p = ex.true_role, ex.pred_role
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    metrics: Dict[str, Dict[str, float]] = {}
    for r in roles:
        t = float(tp[r])
        f_p = float(fp[r])
        f_n = float(fn[r])
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[r] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp[r], "fp": fp[r], "fn": fn[r]}
    return metrics


def print_role_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    if not metrics:
        print("No role metrics available (no labeled examples).")
        return
    print("Per-role metrics (derived from this run):")
    for r in sorted(metrics.keys()):
        m = metrics[r]
        print(
            f"  {r:>10}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  TP={m['tp']} FP={m['fp']} FN={m['fn']}"
        )


def accuracy_vs_prompt_length(examples: List[ClassifiedExample], use_memory: bool) -> None:
    """Report accuracy across prompt-length quartiles (chars), baseline vs with-memory.

    This treats prompt length as a proxy for the amount of context the model sees.
    """
    lens: List[Tuple[int, bool]] = []
    key = "with_memory" if use_memory else "baseline"
    for ex in examples:
        if ex.true_role is None:
            continue
        pstats = ex.prompt_stats or {}
        s = pstats.get(key, {}) or {}
        length = int(s.get("chars", 0))
        correct = ex.pred_role == ex.true_role
        lens.append((length, correct))
    if not lens:
        print("No prompt statistics available for accuracy vs length analysis.")
        return
    lengths = sorted(l for l, _ in lens)
    if not lengths:
        print("No prompt lengths found.")
        return
    # quartiles
    def _q(pos: float) -> int:
        idx = int(pos * (len(lengths) - 1))
        return lengths[idx]

    q1, q2, q3 = _q(0.25), _q(0.5), _q(0.75)
    bins = {
        "short": (0, q1 + 1),
        "medium": (q1 + 1, q2 + 1),
        "long": (q2 + 1, q3 + 1),
        "very_long": (q3 + 1, max(lengths) + 1),
    }
    label = "with_memory" if use_memory else "baseline"
    print(f"Accuracy vs prompt length ({label} prompts; char-based quartiles):")
    for name, (lo, hi) in bins.items():
        total = correct = 0
        for length, is_correct in lens:
            if lo <= length < hi:
                total += 1
                if is_correct:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        print(f"  {name:10} [{lo:4d}, {hi:4d})  N={total:4d}  acc={acc:.3f}")


def accuracy_by_quest_bucket(examples: List[ClassifiedExample]) -> None:
    """Report accuracy on early/middle/late quests.

    Buckets are defined by splitting quest indices into three equal ranges
    within the maximum quest index observed in this run.
    """
    quests = [int(ex.quest) for ex in examples if ex.quest is not None]
    if not quests:
        print("No quest indices available for bucketed analysis.")
        return
    max_q = max(quests)
    if max_q <= 0:
        print("Non-positive quest indices; skipping bucketed analysis.")
        return
    buckets = {
        "early": range(1, max_q // 3 + 1),
        "middle": range(max_q // 3 + 1, 2 * max_q // 3 + 1),
        "late": range(2 * max_q // 3 + 1, max_q + 1),
    }
    print("Accuracy by quest bucket:")
    for name, rng in buckets.items():
        total = correct = 0
        for ex in examples:
            if ex.true_role is None or ex.quest is None:
                continue
            if int(ex.quest) in rng:
                total += 1
                if ex.pred_role == ex.true_role:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        print(f"  {name:6}  N={total:4d}  acc={acc:.3f}")


def load_paired(run_dir: Path) -> List[Tuple[ClassifiedExample, ClassifiedExample]]:
    """Load baseline and current examples and join them by target identifier.

    This enables direct within-round comparison of how memory augmentation
    changes the LLM's decisions.
    """
    base = load_results(run_dir, which="baseline")
    cur = load_results(run_dir, which="current")
    # Use the serialized target dict as a key; in this codebase, entry_id is stable
    # but target may not always contain it, so we fall back to (game_id, quest).
    def _key(ex: ClassifiedExample) -> Tuple[Any, Any, Any]:
        tid = ex.target.get("entry_id") if isinstance(ex.target, dict) else None
        return (tid, ex.game_id, ex.quest)

    base_map = {_key(ex): ex for ex in base}
    pairs: List[Tuple[ClassifiedExample, ClassifiedExample]] = []
    for ex in cur:
        k = _key(ex)
        if k in base_map:
            pairs.append((base_map[k], ex))
    return pairs


def summarize_memory_effect(pairs: List[Tuple[ClassifiedExample, ClassifiedExample]]) -> None:
    """Summarize how often memory changes predictions, and whether it helps.

    Counts how many rounds fall into the following categories:
    - both correct
    - both wrong
    - baseline wrong, memory correct (helpful)
    - baseline correct, memory wrong (harmful)
    """
    if not pairs:
        print("No paired baseline/current examples found; did you run both modes?")
        return
    both_correct = both_wrong = base_wrong_mem_correct = base_correct_mem_wrong = 0
    changed_pred = 0
    for b, c in pairs:
        if b.true_role is None or c.true_role is None:
            continue
        base_ok = b.pred_role == b.true_role
        cur_ok = c.pred_role == c.true_role
        if b.pred_role != c.pred_role:
            changed_pred += 1
        if base_ok and cur_ok:
            both_correct += 1
        elif (not base_ok) and (not cur_ok):
            both_wrong += 1
        elif (not base_ok) and cur_ok:
            base_wrong_mem_correct += 1
        elif base_ok and (not cur_ok):
            base_correct_mem_wrong += 1
    total = both_correct + both_wrong + base_wrong_mem_correct + base_correct_mem_wrong
    print("Effect of memory augmentation (paired baseline vs current):")
    print(f"  total paired examples: {total}")
    print(f"  both correct            : {both_correct}")
    print(f"  both wrong              : {both_wrong}")
    print(f"  baseline wrong -> mem ok: {base_wrong_mem_correct}")
    print(f"  baseline ok   -> mem wrong: {base_correct_mem_wrong}")
    if total > 0:
        net = base_wrong_mem_correct - base_correct_mem_wrong
        print(f"  net helpful (helpful - harmful): {net} (changed preds: {changed_pred})")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explainability analysis for LLM proposer-role predictions.")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to a single run directory (e.g., outputs/eval/.../runs/run_1)")
    ap.add_argument("--which", type=str, default="current", choices=["current", "baseline"], help="Analyze 'current' (with memory) or 'baseline' results.")
    ap.add_argument(
        "--mode",
        type=str,
        default="top_confusions",
        choices=[
            "top_confusions",
            "examples",
            "heatmap",
            "role_metrics",
            "prompt_length",
            "quest_buckets",
            "compare_baseline",
        ],
        help="What to compute/visualize.",
    )
    ap.add_argument("--top_n", type=int, default=10, help="Number of top confusion pairs to print (top_confusions mode).")
    ap.add_argument("--true_role", type=str, default=None, help="True role to filter on (examples mode).")
    ap.add_argument("--pred_role", type=str, default=None, help="Predicted role to filter on (examples mode).")
    ap.add_argument("--max_examples", type=int, default=5, help="Maximum number of examples to show (examples mode).")
    ap.add_argument("--heatmap_html", type=str, default="confusion_explain.html", help="Output HTML filename for heatmap (heatmap mode).")
    ap.add_argument("--use_memory", action="store_true", help="In prompt_length mode, analyze with_memory prompts instead of baseline.")
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
    elif args.mode == "role_metrics":
        metrics = compute_role_metrics(examples)
        print_role_metrics(metrics)
    elif args.mode == "prompt_length":
        # default to baseline unless explicitly asked for with_memory
        accuracy_vs_prompt_length(examples, use_memory=bool(args.use_memory))
    elif args.mode == "quest_buckets":
        accuracy_by_quest_bucket(examples)
    elif args.mode == "compare_baseline":
        pairs = load_paired(run_dir)
        summarize_memory_effect(pairs)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
