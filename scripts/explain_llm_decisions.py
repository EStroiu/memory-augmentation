#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.offline import plot as plotly_offline_plot

try:
    from scripts.metrics_utils import viz_confusion_matrix
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.metrics_utils import viz_confusion_matrix

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
    vector_memory: Dict[str, Any]
    llm_processed: Dict[str, Any]


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
                vector_memory=r.get("vector_memory", {}) or {},
                llm_processed=r.get("llm_processed", {}) or {},
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
    """Create a confusion heatmap using the shared metrics_utils helper."""
    roles = sorted({tr for tr, _ in counts.keys()} | {pr for _, pr in counts.keys()})
    if not roles:
        print("No roles to visualize.")
        return
    n = len(roles)
    idx = {r: i for i, r in enumerate(roles)}
    M = np.zeros((n, n), dtype=float)
    for (tr, pr), c in counts.items():
        if tr not in idx or pr not in idx:
            continue
        M[idx[tr], idx[pr]] += c
    # Convert back to the dict[true][pred] shape expected by viz_confusion_matrix
    confusion: Dict[str, Dict[str, float]] = {tr: {pr: 0.0 for pr in roles} for tr in roles}
    for i, tr in enumerate(roles):
        for j, pr in enumerate(roles):
            confusion[tr][pr] = float(M[i, j])
    viz_confusion_matrix(out_html.parent, "LLM confusion heatmap (true vs predicted)", roles, confusion, None, out_html.name)
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


def summarize_belief_effect(
    pairs: List[Tuple[ClassifiedExample, ClassifiedExample]],
    max_examples: int = 5,
) -> None:
    """Summarize how belief/vector memory correlates with prediction changes.

    Prints a brief delta summary and a few illustrative examples with
    baseline vs current predictions and any vector-memory preview available.
    """
    if not pairs:
        print("No paired baseline/current examples found; did you run both modes?")
        return

    changed = 0
    helpful = 0
    harmful = 0
    for b, c in pairs:
        if b.true_role is None or c.true_role is None:
            continue
        if b.pred_role != c.pred_role:
            changed += 1
        base_ok = b.pred_role == b.true_role
        cur_ok = c.pred_role == c.true_role
        if (not base_ok) and cur_ok:
            helpful += 1
        elif base_ok and (not cur_ok):
            harmful += 1

    total = helpful + harmful
    print("Belief/vector-memory effect summary:")
    print(f"  changed predictions: {changed}")
    print(f"  helpful changes    : {helpful}")
    print(f"  harmful changes    : {harmful}")
    if total > 0:
        net = helpful - harmful
        print(f"  net helpful        : {net}")

    print("\nExamples with vector-memory preview (if available):")
    shown = 0
    for b, c in pairs:
        if shown >= max_examples:
            break
        if b.true_role is None or c.true_role is None:
            continue
        # prioritize cases where prediction changes
        if b.pred_role == c.pred_role:
            continue
        vm_preview = c.vector_memory.get("preview") if isinstance(c.vector_memory, dict) else None
        print("-" * 80)
        print(f"game_id: {c.game_id}  quest: {c.quest}")
        print(f"true_role: {c.true_role}")
        print(f"baseline_pred: {b.pred_role}  current_pred: {c.pred_role}")
        if vm_preview:
            print(f"vector_memory_preview: {vm_preview}")
        else:
            print("vector_memory_preview: <none>")
        shown += 1


def _find_experiment_dir(path: Path) -> Path:
    """Resolve an experiment directory that contains a `runs/` folder.

    Accepts either:
    - experiment directory (`.../outputs/eval/<exp_id>`)
    - runs directory (`.../outputs/eval/<exp_id>/runs`)
    - single run directory (`.../outputs/eval/<exp_id>/runs/run_1`)
    """
    p = path.resolve()
    if (p / "runs").exists():
        return p
    if p.name == "runs" and p.parent.exists():
        return p.parent
    if p.name.startswith("run_") and p.parent.name == "runs" and p.parent.parent.exists():
        return p.parent.parent
    raise FileNotFoundError(f"Could not resolve experiment dir with runs/: {path}")


def _find_single_run_dir(path: Path) -> Path:
    """Resolve a single `run_*` directory from experiment/runs/run input."""
    p = path.resolve()
    if p.name.startswith("run_") and p.is_dir():
        return p
    exp_dir = _find_experiment_dir(p)
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found under: {exp_dir}")
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* subdirectories found under: {runs_dir}")
    return run_dirs[0]


def _collect_metric_from_runs(experiment_dir: Path, which: str, metric: str) -> List[float]:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return []
    values: List[float] = []
    for run_subdir in sorted(runs_dir.iterdir()):
        if not run_subdir.is_dir() or not run_subdir.name.startswith("run_"):
            continue
        agg_path = run_subdir / f"aggregate_{which}.json"
        if not agg_path.exists():
            continue
        try:
            with agg_path.open("r", encoding="utf-8") as f:
                agg = json.load(f)
            val = agg.get(metric)
            if val is None:
                continue
            values.append(float(val))
        except Exception:
            continue
    return values


def _collect_role_metric_from_runs(
    experiment_dir: Path,
    which: str,
    role_metric: str,
) -> Dict[str, List[float]]:
    """Collect per-role metric arrays from per-run aggregate files.

    Returns a mapping role -> list of metric values across runs.
    """
    runs_dir = experiment_dir / "runs"
    out: Dict[str, List[float]] = {}
    if not runs_dir.exists():
        return out
    for run_subdir in sorted(runs_dir.iterdir()):
        if not run_subdir.is_dir() or not run_subdir.name.startswith("run_"):
            continue
        agg_path = run_subdir / f"aggregate_{which}.json"
        if not agg_path.exists():
            continue
        try:
            with agg_path.open("r", encoding="utf-8") as f:
                agg = json.load(f)
            by_role = agg.get("clf_by_role", {}) or {}
            for role, m in by_role.items():
                if not isinstance(m, dict):
                    continue
                val = m.get(role_metric)
                if val is None:
                    continue
                out.setdefault(str(role), []).append(float(val))
        except Exception:
            continue
    return out


def plot_baseline_vs_current_performance(
    experiment_dir: Path,
    metric: str = "clf_micro_f1",
    out_html: str = "performance_compare.html",
) -> None:
    """Plot current vs baseline mean performance with std-dev error bars.

    The plot is built from per-run aggregate files under `runs/run_*/aggregate_*.json`.
    If baseline files are absent, the chart will include only current.
    """
    current_vals = _collect_metric_from_runs(experiment_dir, which="current", metric=metric)
    baseline_vals = _collect_metric_from_runs(experiment_dir, which="baseline", metric=metric)

    if not current_vals and not baseline_vals:
        print(
            "No run-level metrics found for plotting. "
            f"Expected files like runs/run_*/aggregate_current.json with key '{metric}'."
        )
        return

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []

    if baseline_vals:
        labels.append("baseline")
        means.append(float(np.mean(baseline_vals)))
        stds.append(float(np.std(baseline_vals)))
        counts.append(len(baseline_vals))
    if current_vals:
        labels.append("current")
        means.append(float(np.mean(current_vals)))
        stds.append(float(np.std(current_vals)))
        counts.append(len(current_vals))

    text = [f"N={n}<br>mean={m:.4f}<br>std={s:.4f}" for n, m, s in zip(counts, means, stds)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
                text=text,
                textposition="outside",
                hovertemplate="%{x}<br>mean=%{y:.4f}<br>%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Baseline vs current performance ({metric})",
        xaxis_title="Condition",
        yaxis_title=metric,
        yaxis=dict(range=[0.0, 1.0]),
    )

    out_path = experiment_dir / out_html
    plotly_offline_plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs="cdn")
    print(f"Saved performance comparison plot to: {out_path}")
    for label, n, m, s in zip(labels, counts, means, stds):
        print(f"  {label:>8}: N={n:>3d} mean={m:.6f} std={s:.6f}")


def plot_baseline_vs_current_role_performance(
    experiment_dir: Path,
    role_metric: str = "f1",
    out_html: str = "performance_compare_roles.html",
) -> None:
    """Plot per-role baseline vs current with std-dev error bars across runs."""
    baseline_map = _collect_role_metric_from_runs(experiment_dir, which="baseline", role_metric=role_metric)
    current_map = _collect_role_metric_from_runs(experiment_dir, which="current", role_metric=role_metric)

    roles = sorted(set(baseline_map.keys()) | set(current_map.keys()))
    if not roles:
        print(
            "No role-level metrics found for plotting. "
            "Expected clf_by_role in runs/run_*/aggregate_{baseline,current}.json."
        )
        return

    baseline_means = [float(np.mean(baseline_map.get(r, [0.0]))) for r in roles]
    baseline_stds = [float(np.std(baseline_map.get(r, [0.0]))) for r in roles]
    current_means = [float(np.mean(current_map.get(r, [0.0]))) for r in roles]
    current_stds = [float(np.std(current_map.get(r, [0.0]))) for r in roles]

    baseline_text = [
        f"N={len(baseline_map.get(r, []))}<br>mean={m:.4f}<br>std={s:.4f}"
        for r, m, s in zip(roles, baseline_means, baseline_stds)
    ]
    current_text = [
        f"N={len(current_map.get(r, []))}<br>mean={m:.4f}<br>std={s:.4f}"
        for r, m, s in zip(roles, current_means, current_stds)
    ]

    fig = go.Figure()
    if any(len(baseline_map.get(r, [])) > 0 for r in roles):
        fig.add_trace(
            go.Bar(
                name="baseline",
                x=roles,
                y=baseline_means,
                error_y=dict(type="data", array=baseline_stds, visible=True),
                text=baseline_text,
                hovertemplate="role=%{x}<br>baseline mean=%{y:.4f}<br>%{text}<extra></extra>",
            )
        )
    if any(len(current_map.get(r, [])) > 0 for r in roles):
        fig.add_trace(
            go.Bar(
                name="current",
                x=roles,
                y=current_means,
                error_y=dict(type="data", array=current_stds, visible=True),
                text=current_text,
                hovertemplate="role=%{x}<br>current mean=%{y:.4f}<br>%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Per-role baseline vs current ({role_metric})",
        barmode="group",
        xaxis_title="Role",
        yaxis_title=role_metric,
        yaxis=dict(range=[0.0, 1.0]),
    )

    out_path = experiment_dir / out_html
    plotly_offline_plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs="cdn")
    print(f"Saved per-role performance comparison plot to: {out_path}")


def run_full_explainability(
    run_dir: Path,
    which: str,
    top_n: int,
    max_examples: int,
    heatmap_html: str,
    use_memory: bool,
    compare_metric: str,
    compare_html: str,
    compare_roles_html: str,
    role_metric: str,
) -> None:
    """Run a complete explainability pass for one run + experiment-level comparison."""
    examples = load_results(run_dir, which=which)
    counts = build_confusion_counts(examples)

    print("\n=== Top confusions ===")
    print_top_confusions(counts, top_n)

    print("\n=== Role metrics ===")
    metrics = compute_role_metrics(examples)
    print_role_metrics(metrics)

    print("\n=== Accuracy vs prompt length ===")
    accuracy_vs_prompt_length(examples, use_memory=use_memory)

    print("\n=== Accuracy by quest bucket ===")
    accuracy_by_quest_bucket(examples)

    print("\n=== Paired baseline/current comparison ===")
    pairs = load_paired(run_dir)
    summarize_memory_effect(pairs)

    print("\n=== Belief/vector memory effect ===")
    summarize_belief_effect(pairs, max_examples=max_examples)

    print("\n=== Confusion heatmap ===")
    viz_confusion_heatmap(counts, run_dir / heatmap_html)

    print("\n=== Baseline vs current performance plot (error bars) ===")
    experiment_dir = _find_experiment_dir(run_dir)
    plot_baseline_vs_current_performance(experiment_dir, metric=compare_metric, out_html=compare_html)

    print("\n=== Per-role baseline vs current plot (error bars) ===")
    plot_baseline_vs_current_role_performance(
        experiment_dir,
        role_metric=role_metric,
        out_html=compare_roles_html,
    )


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
            "compare_beliefs",
            "compare_performance_plot",
            "compare_role_performance_plot",
            "full_eval",
        ],
        help="What to compute/visualize.",
    )
    ap.add_argument("--top_n", type=int, default=10, help="Number of top confusion pairs to print (top_confusions mode).")
    ap.add_argument("--true_role", type=str, default=None, help="True role to filter on (examples mode).")
    ap.add_argument("--pred_role", type=str, default=None, help="Predicted role to filter on (examples mode).")
    ap.add_argument("--max_examples", type=int, default=5, help="Maximum number of examples to show (examples mode).")
    ap.add_argument("--heatmap_html", type=str, default="confusion_explain.html", help="Output HTML filename for heatmap (heatmap mode).")
    ap.add_argument("--use_memory", action="store_true", help="In prompt_length mode, analyze with_memory prompts instead of baseline.")
    ap.add_argument(
        "--compare_metric",
        type=str,
        default="clf_micro_f1",
        help="Metric key to compare in compare_performance_plot/full_eval (read from aggregate_*.json).",
    )
    ap.add_argument(
        "--compare_html",
        type=str,
        default="performance_compare.html",
        help="Output HTML filename for compare_performance_plot/full_eval.",
    )
    ap.add_argument(
        "--role_metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Per-role metric to plot in compare_role_performance_plot/full_eval.",
    )
    ap.add_argument(
        "--compare_roles_html",
        type=str,
        default="performance_compare_roles.html",
        help="Output HTML filename for compare_role_performance_plot/full_eval.",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        return 1

    if args.mode == "compare_performance_plot":
        try:
            experiment_dir = _find_experiment_dir(run_dir)
        except Exception as e:
            print(f"Failed to resolve experiment dir: {e}")
            return 1
        plot_baseline_vs_current_performance(
            experiment_dir,
            metric=str(args.compare_metric),
            out_html=str(args.compare_html),
        )
        return 0

    if args.mode == "compare_role_performance_plot":
        try:
            experiment_dir = _find_experiment_dir(run_dir)
        except Exception as e:
            print(f"Failed to resolve experiment dir: {e}")
            return 1
        plot_baseline_vs_current_role_performance(
            experiment_dir,
            role_metric=str(args.role_metric),
            out_html=str(args.compare_roles_html),
        )
        return 0

    if args.mode == "full_eval":
        try:
            selected_run_dir = _find_single_run_dir(run_dir)
        except Exception as e:
            print(f"Failed to resolve run directory for full_eval: {e}")
            return 1
        run_full_explainability(
            run_dir=selected_run_dir,
            which=str(args.which),
            top_n=int(args.top_n),
            max_examples=int(args.max_examples),
            heatmap_html=str(args.heatmap_html),
            use_memory=bool(args.use_memory),
            compare_metric=str(args.compare_metric),
            compare_html=str(args.compare_html),
            compare_roles_html=str(args.compare_roles_html),
            role_metric=str(args.role_metric),
        )
        return 0

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
    elif args.mode == "compare_beliefs":
        pairs = load_paired(run_dir)
        summarize_belief_effect(pairs, max_examples=int(args.max_examples))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
