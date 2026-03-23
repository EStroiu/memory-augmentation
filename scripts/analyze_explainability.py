#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go


ROLE_ORDER = ["assassin", "merlin", "morgana", "percival", "servant-1", "servant-2"]
EVIL_ROLES = {"assassin", "morgana"}
GOOD_ROLES = {"percival", "servant-1", "servant-2"}


@dataclass
class Example:
    true_role: str | None
    pred_role: str | None
    game_id: str | None
    quest: int | None
    llm_processed: Dict[str, Any]
    prompt_chars: int | None


def _configure_plot_style() -> None:
    return


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_examples(result_file: Path) -> List[Example]:
    data = load_json(result_file)
    out: List[Example] = []
    for row in data:
        target = row.get("target", {}) or {}
        cls = row.get("classification", {}) or {}
        prompt_stats = row.get("prompt_stats", {}) or {}
        with_mem = prompt_stats.get("with_memory", {}) or {}
        quest_val = target.get("quest")
        try:
            q = int(quest_val) if quest_val is not None else None
        except Exception:
            q = None

        out.append(
            Example(
                true_role=cls.get("true_role"),
                pred_role=cls.get("pred_role"),
                game_id=target.get("game_id"),
                quest=q,
                llm_processed=row.get("llm_processed", {}) or {},
                prompt_chars=with_mem.get("chars"),
            )
        )
    return out


def load_all_runs(exp_dir: Path, mode: str) -> List[Example]:
    examples: List[Example] = []
    runs_dir = exp_dir / "runs"
    run_dirs = sorted([p for p in runs_dir.glob("run_*") if p.is_dir()], key=lambda p: p.name)
    fname = "results_baseline.json" if mode == "baseline" else "results_current.json"
    for rd in run_dirs:
        f = rd / fname
        if f.exists():
            examples.extend(read_examples(f))
    return examples


def load_runs_grouped(exp_dir: Path, mode: str) -> List[List[Example]]:
    grouped: List[List[Example]] = []
    runs_dir = exp_dir / "runs"
    run_dirs = sorted([p for p in runs_dir.glob("run_*") if p.is_dir()], key=lambda p: p.name)
    fname = "results_baseline.json" if mode == "baseline" else "results_current.json"
    for rd in run_dirs:
        f = rd / fname
        if f.exists():
            grouped.append(read_examples(f))
    return grouped


def count_confusion(examples: Iterable[Example]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {tr: {pr: 0 for pr in ROLE_ORDER} for tr in ROLE_ORDER}
    for ex in examples:
        if ex.true_role in ROLE_ORDER and ex.pred_role in ROLE_ORDER:
            matrix[ex.true_role][ex.pred_role] += 1
    return matrix


def role_count_vector(examples: Iterable[Example], source: str) -> np.ndarray:
    counts = Counter()
    for ex in examples:
        role = ex.true_role if source == "true" else ex.pred_role
        if role in ROLE_ORDER:
            counts[str(role)] += 1
    return np.array([float(counts.get(r, 0)) for r in ROLE_ORDER], dtype=float)


def role_share_vector(examples: Iterable[Example], source: str) -> np.ndarray:
    vec = role_count_vector(examples, source=source)
    s = vec.sum()
    return vec / s if s > 0 else vec


def assassin_bucket_vector(examples: Iterable[Example]) -> np.ndarray:
    by_q = _bucket_by_quest(examples, "assassin")
    return np.array([float(by_q[k]) for k in ["q1", "q2", "q3", "q4+"]], dtype=float)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def per_class_f1(true_labels: List[str], pred_labels: List[str], labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == lab and p != lab)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        out[lab] = f1
    return out


def role_to_three_class(role: str | None) -> str | None:
    if role is None:
        return None
    if role == "merlin":
        return "merlin"
    if role in EVIL_ROLES:
        return "evil"
    if role in GOOD_ROLES:
        return "good"
    return None


def three_class_metrics(examples: Iterable[Example]) -> Dict[str, Any]:
    labels = ["good", "evil", "merlin"]
    y_true: List[str] = []
    y_pred: List[str] = []

    for ex in examples:
        t = role_to_three_class(ex.true_role)
        p = role_to_three_class(ex.pred_role)
        # Keep all labeled targets. Unmapped/null predictions are counted as errors.
        if t is None:
            continue
        if p is None:
            p = "__other__"
        y_true.append(t)
        y_pred.append(p)

    per_f1 = per_class_f1(y_true, y_pred, labels)
    tp_micro = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    micro_f1 = safe_div(tp_micro, len(y_true))
    macro_f1 = float(np.mean([per_f1[l] for l in labels])) if labels else 0.0

    return {
        "support": len(y_true),
        "f1_by_class": per_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def plot_distribution_comparison(
    true_means: np.ndarray,
    true_stds: np.ndarray,
    pred_means: np.ndarray,
    pred_stds: np.ndarray,
    out_pdf: Path,
    title: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=true_means,
            y=ROLE_ORDER,
            orientation="h",
            name="True",
            marker_color="#4C78A8",
            error_x={"type": "data", "array": true_stds, "visible": True},
        )
    )
    fig.add_trace(
        go.Bar(
            x=pred_means,
            y=ROLE_ORDER,
            orientation="h",
            name="Predicted",
            marker_color="#F58518",
            error_x={"type": "data", "array": pred_stds, "visible": True},
        )
    )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Count (mean across runs)",
        yaxis_title="Role",
        template="plotly_white",
        width=1050,
        height=620,
        margin={"l": 120, "r": 40, "t": 70, "b": 70},
        font={"size": 16},
        legend={"font": {"size": 13}},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_confusion_heatmap(conf: Dict[str, Dict[str, int]], out_pdf: Path, title: str) -> None:
    m = np.array([[conf[tr][pr] for pr in ROLE_ORDER] for tr in ROLE_ORDER], dtype=float)
    row_sums = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, np.where(row_sums == 0, 1.0, row_sums))

    text = [[f"{norm[i, j] * 100:.1f}%<br>({int(m[i, j])})" for j in range(len(ROLE_ORDER))] for i in range(len(ROLE_ORDER))]
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=norm,
                x=ROLE_ORDER,
                y=ROLE_ORDER,
                colorscale="Blues",
                zmin=0.0,
                zmax=1.0,
                colorbar={"title": "Row-normalized proportion"},
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="True=%{y}<br>Pred=%{x}<br>Rate=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted role",
        yaxis_title="True role",
        template="plotly_white",
        width=1240,
        height=1060,
        margin={"l": 130, "r": 70, "t": 80, "b": 90},
        font={"size": 14},
    )
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(autorange="reversed")
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_three_class_f1(
    metrics_by_condition: Dict[str, Dict[str, Any]],
    per_run_three_class: Dict[str, List[Dict[str, Any]]],
    out_pdf: Path,
) -> None:
    classes = ["good", "evil", "merlin"]
    conds = list(metrics_by_condition.keys())
    fig = go.Figure()
    for idx, cond in enumerate(conds):
        vals = np.array([metrics_by_condition[cond]["f1_by_class"].get(c, 0.0) for c in classes], dtype=float)
        runs = per_run_three_class.get(cond, [])
        if runs:
            arr = np.array([[r["f1_by_class"].get(c, 0.0) for c in classes] for r in runs], dtype=float)
            errs = arr.std(axis=0)
        else:
            errs = np.zeros_like(vals)
        fig.add_trace(
            go.Bar(
                x=vals,
                y=classes,
                orientation="h",
                name=cond,
                error_x={"type": "data", "array": errs, "visible": True},
            )
        )
    fig.update_layout(
        barmode="group",
        title="3-Class F1 (good / evil / merlin)",
        xaxis_title="F1",
        yaxis_title="Grouped Class",
        template="plotly_white",
        width=1060,
        height=650,
        margin={"l": 130, "r": 60, "t": 80, "b": 80},
        xaxis={"range": [0.0, 1.0]},
        font={"size": 16},
        legend={"font": {"size": 13}},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def _top_share(pred_counts: Counter[str], role: str) -> float:
    tot = sum(pred_counts.values())
    return safe_div(pred_counts.get(role, 0), tot)


def _true_role_breakdown_for_pred(examples: Iterable[Example], role: str) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for ex in examples:
        if ex.pred_role == role and ex.true_role is not None:
            c[ex.true_role] += 1
    return dict(c)


def _bucket_by_quest(examples: Iterable[Example], role: str) -> Dict[str, int]:
    out = {"q1": 0, "q2": 0, "q3": 0, "q4+": 0}
    for ex in examples:
        if ex.pred_role != role or ex.quest is None:
            continue
        if ex.quest <= 1:
            out["q1"] += 1
        elif ex.quest == 2:
            out["q2"] += 1
        elif ex.quest == 3:
            out["q3"] += 1
        else:
            out["q4+"] += 1
    return out


def _extract_decision_stats(examples: Iterable[Example]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    used_repair = 0
    proposer_override = 0
    fallback_reasons: Counter[str] = Counter()
    decision_tokens: Counter[str] = Counter()
    n = 0

    for ex in examples:
        p = ex.llm_processed or {}
        n += 1
        if p.get("used_repair") is True:
            used_repair += 1
        if p.get("proposer_override_applied") is True:
            proposer_override += 1
        fr = p.get("fallback_reason")
        if isinstance(fr, str) and fr:
            fallback_reasons[fr] += 1
        for d in p.get("decision_path", []) or []:
            decision_tokens[str(d)] += 1

    stats["n_examples"] = n
    stats["used_repair_rate"] = safe_div(used_repair, n)
    stats["proposer_override_rate"] = safe_div(proposer_override, n)
    stats["fallback_reasons"] = dict(fallback_reasons)
    stats["decision_path_tokens"] = dict(decision_tokens)
    return stats


def plot_assassin_diagnostics(means: np.ndarray, stds: np.ndarray, out_pdf: Path) -> None:
    labels = ["q1", "q2", "q3", "q4+"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=means,
                y=labels,
                orientation="h",
                marker_color="#E45756",
                error_x={"type": "data", "array": stds, "visible": True},
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title="Belief+Social: Assassin Predictions by Quest Bucket",
        xaxis_title="Predicted assassin count (mean across runs)",
        yaxis_title="Quest Bucket",
        template="plotly_white",
        width=900,
        height=520,
        margin={"l": 110, "r": 50, "t": 75, "b": 70},
        font={"size": 15},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_micro_f1_boxplot(per_run_three_class: Dict[str, List[Dict[str, Any]]], out_pdf: Path) -> None:
    conds = list(per_run_three_class.keys())
    data = []
    labels = []
    for c in conds:
        vals = [float(m.get("micro_f1", 0.0)) for m in per_run_three_class.get(c, [])]
        if vals:
            data.append(vals)
            labels.append(c)

    if not data:
        return

    fig = go.Figure()
    for label, vals in zip(labels, data):
        fig.add_trace(
            go.Box(
                x=vals,
                y=[label] * len(vals),
                orientation="h",
                name=label,
                boxpoints=False,
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Run-Level 3-Class Micro-F1 Distribution",
        xaxis_title="3-class micro-F1",
        yaxis_title="Condition",
        template="plotly_white",
        width=1080,
        height=620,
        margin={"l": 200, "r": 50, "t": 75, "b": 75},
        font={"size": 15},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_role_bias_shift(
    baseline_pred_share: np.ndarray,
    belief_social_pred_share: np.ndarray,
    baseline_std: np.ndarray,
    belief_social_std: np.ndarray,
    out_pdf: Path,
) -> None:
    diff = belief_social_pred_share - baseline_pred_share
    err = np.sqrt(np.square(baseline_std) + np.square(belief_social_std))
    colors = ["#D62728" if d > 0 else "#1F77B4" for d in diff]
    fig = go.Figure(
        data=[
            go.Bar(
                x=diff,
                y=ROLE_ORDER,
                orientation="h",
                marker_color=colors,
                error_x={"type": "data", "array": err, "visible": True},
                showlegend=False,
            )
        ]
    )
    fig.add_vline(x=0.0, line_width=1.2, line_color="black")
    fig.update_layout(
        title="Prediction-Bias Shift by Role",
        xaxis_title="Predicted role share shift (belief+social - baseline)",
        yaxis_title="Role",
        template="plotly_white",
        width=1040,
        height=620,
        margin={"l": 140, "r": 50, "t": 75, "b": 80},
        font={"size": 15},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def write_report(
    out_path: Path,
    baseline_examples: List[Example],
    focus_sections: List[Tuple[str, List[Example]]],
    three_class: Dict[str, Dict[str, Any]],
) -> None:
    base_pred = Counter(ex.pred_role for ex in baseline_examples if ex.pred_role)

    base_true_break = _true_role_breakdown_for_pred(baseline_examples, "servant-1")

    lines: List[str] = []
    lines.append("# Explainability Analysis Report")
    lines.append("")
    lines.append("## 1) Why baseline focuses on servant-1")
    lines.append(f"- Share of servant-1 predictions: {(_top_share(base_pred, 'servant-1')*100):.2f}%")
    lines.append("- True-role composition among predictions labeled servant-1:")
    for r in ROLE_ORDER:
        lines.append(f"  - {r}: {base_true_break.get(r, 0)}")
    lines.append("- Interpretation:")
    lines.append("  - Baseline tends to collapse toward a majority safe class when evidence is ambiguous.")
    lines.append("  - This is consistent with high confusion into servant classes in aggregate confusion matrices.")
    lines.append("")
    lines.append("## 2) Focus-technique behavior")
    for focus_label, focus_examples in focus_sections:
        focus_pred = Counter(ex.pred_role for ex in focus_examples if ex.pred_role)
        focus_top_role = max(focus_pred.items(), key=lambda x: x[1])[0] if focus_pred else "assassin"
        focus_true_break = _true_role_breakdown_for_pred(focus_examples, focus_top_role)
        focus_decisions = _extract_decision_stats(focus_examples)

        lines.append(f"### {focus_label}")
        lines.append(f"- Top predicted role: {focus_top_role}")
        lines.append(f"- Share of top-role predictions: {(_top_share(focus_pred, focus_top_role)*100):.2f}%")
        lines.append(f"- True-role composition among predictions labeled {focus_top_role}:")
        for r in ROLE_ORDER:
            lines.append(f"  - {r}: {focus_true_break.get(r, 0)}")
        lines.append("- LLM post-processing/decision-path diagnostics:")
        lines.append(f"  - used_repair_rate: {focus_decisions['used_repair_rate']:.3f}")
        lines.append(f"  - proposer_override_rate: {focus_decisions['proposer_override_rate']:.3f}")
        lines.append(f"  - fallback_reasons: {focus_decisions['fallback_reasons']}")
        lines.append("- Interpretation:")
        lines.append("  - High repair usage and strong post-processing influence can amplify one role decision path.")
        lines.append("  - Read concentrated predictions as a system-level behavior (prompt + repair + proposer mapping), not only raw model intent.")
    lines.append("")
    lines.append("## 3) F1 for good/evil/merlin")
    for cond, met in three_class.items():
        lines.append(f"### {cond}")
        lines.append(f"- support: {met['support']}")
        lines.append(f"- micro_f1: {met['micro_f1']:.4f}")
        lines.append(f"- macro_f1: {met['macro_f1']:.4f}")
        for c in ["good", "evil", "merlin"]:
            lines.append(f"  - F1({c}): {met['f1_by_class'].get(c, 0.0):.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is computed from per-instance run outputs across all runs in each condition directory.")
    lines.append("- Plot PDFs are generated with enlarged fonts and constrained layouts for paper readability.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_condition_bundle(
    cond_label: str,
    cond_slug: str,
    cond_examples: List[Example],
    cond_runs: List[List[Example]],
    outdir: Path,
) -> None:
    true_arr = np.array([role_count_vector(run, source="true") for run in cond_runs], dtype=float)
    pred_arr = np.array([role_count_vector(run, source="pred") for run in cond_runs], dtype=float)

    plot_distribution_comparison(
        true_arr.mean(axis=0),
        true_arr.std(axis=0),
        pred_arr.mean(axis=0),
        pred_arr.std(axis=0),
        outdir / f"{cond_slug}_true_vs_pred_distribution.pdf",
        f"{cond_label}: True vs Predicted Role Distribution",
    )

    conf = count_confusion(cond_examples)
    plot_confusion_heatmap(
        conf,
        outdir / f"{cond_slug}_confusion_heatmap.pdf",
        f"{cond_label} Confusion Matrix (Row-normalized)",
    )

    assassin_bucket_arr = np.array([assassin_bucket_vector(run) for run in cond_runs], dtype=float)
    plot_assassin_diagnostics(
        assassin_bucket_arr.mean(axis=0),
        assassin_bucket_arr.std(axis=0),
        outdir / f"{cond_slug}_assassin_by_quest_bucket.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="In-depth explainability analysis for Avalon role-inference experiments.")
    parser.add_argument(
        "--baseline-exp",
        type=Path,
        default=Path("outputs/eval/06-03-2026_11-09-22_mem-template_exp-baseline_full"),
        help="Experiment directory for baseline-only run set.",
    )
    parser.add_argument(
        "--vector-exp",
        type=Path,
        default=Path("outputs/eval/06-03-2026_12-36-08_mem-template_exp-custom"),
        help="Experiment directory for vector_memory condition.",
    )
    parser.add_argument(
        "--belief-exp",
        type=Path,
        default=Path("outputs/eval/08-03-2026_16-39-46_mem-template_exp-custom"),
        help="Experiment directory for belief_vector condition.",
    )
    parser.add_argument(
        "--belief-social-exp",
        type=Path,
        default=Path("outputs/eval/08-03-2026_21-57-07_mem-template_exp-custom"),
        help="Experiment directory for belief_vector+social condition.",
    )
    parser.add_argument(
        "--self-note-exp",
        type=Path,
        default=None,
        help="Optional experiment directory for llm_self_note condition.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/analysis/explainability"),
        help="Directory where report and PDF plots are saved.",
    )
    args = parser.parse_args()

    _configure_plot_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    baseline_examples = load_all_runs(args.baseline_exp, mode="baseline")
    vector_examples = load_all_runs(args.vector_exp, mode="current")
    belief_examples = load_all_runs(args.belief_exp, mode="current")
    belief_social_examples = load_all_runs(args.belief_social_exp, mode="current")
    self_note_examples = load_all_runs(args.self_note_exp, mode="current") if args.self_note_exp else []

    baseline_runs = load_runs_grouped(args.baseline_exp, mode="baseline")
    vector_runs = load_runs_grouped(args.vector_exp, mode="current")
    belief_runs = load_runs_grouped(args.belief_exp, mode="current")
    belief_social_runs = load_runs_grouped(args.belief_social_exp, mode="current")
    self_note_runs = load_runs_grouped(args.self_note_exp, mode="current") if args.self_note_exp else []

    # Q1: baseline servant-1 collapse
    base_true_arr = np.array([role_count_vector(run, source="true") for run in baseline_runs], dtype=float)
    base_pred_arr = np.array([role_count_vector(run, source="pred") for run in baseline_runs], dtype=float)
    plot_distribution_comparison(
        base_true_arr.mean(axis=0),
        base_true_arr.std(axis=0),
        base_pred_arr.mean(axis=0),
        base_pred_arr.std(axis=0),
        args.outdir / "baseline_true_vs_pred_distribution.pdf",
        "Baseline: True vs Predicted Role Distribution",
    )
    baseline_conf = count_confusion(baseline_examples)
    plot_confusion_heatmap(
        baseline_conf,
        args.outdir / "baseline_confusion_heatmap.pdf",
        "Baseline Confusion Matrix (Row-normalized)",
    )

    # Q2: concentration diagnostics for all focus techniques
    focus_conditions: List[Tuple[str, str, List[Example], List[List[Example]]]] = [
        ("belief_vector+social", "belief_social", belief_social_examples, belief_social_runs),
    ]
    if args.self_note_exp:
        focus_conditions.append(("llm_self_note", "self_note", self_note_examples, self_note_runs))

    for cond_label, cond_slug, cond_examples, cond_runs in focus_conditions:
        _plot_condition_bundle(cond_label, cond_slug, cond_examples, cond_runs, args.outdir)

    # Q3: 3-class good/evil/merlin F1 for all focused conditions
    conditions: List[Tuple[str, List[Example], List[List[Example]]]] = [
        ("baseline", baseline_examples, baseline_runs),
        ("vector_memory", vector_examples, vector_runs),
        ("belief_vector", belief_examples, belief_runs),
        ("belief_vector+social", belief_social_examples, belief_social_runs),
    ]
    if args.self_note_exp:
        conditions.append(("llm_self_note", self_note_examples, self_note_runs))

    three_class = {name: three_class_metrics(exs) for name, exs, _ in conditions}
    per_run_three_class = {name: [three_class_metrics(run) for run in runs] for name, _, runs in conditions}
    plot_three_class_f1(three_class, per_run_three_class, args.outdir / "three_class_f1_comparison.pdf")
    plot_micro_f1_boxplot(per_run_three_class, args.outdir / "three_class_micro_f1_boxplot.pdf")

    base_share_arr = np.array([role_share_vector(run, source="pred") for run in baseline_runs], dtype=float)
    bs_share_arr = np.array([role_share_vector(run, source="pred") for run in belief_social_runs], dtype=float)
    plot_role_bias_shift(
        base_share_arr.mean(axis=0),
        bs_share_arr.mean(axis=0),
        base_share_arr.std(axis=0),
        bs_share_arr.std(axis=0),
        args.outdir / "role_bias_shift_baseline_vs_belief_social.pdf",
    )
    if args.self_note_exp:
        sn_share_arr = np.array([role_share_vector(run, source="pred") for run in self_note_runs], dtype=float)
        plot_role_bias_shift(
            base_share_arr.mean(axis=0),
            sn_share_arr.mean(axis=0),
            base_share_arr.std(axis=0),
            sn_share_arr.std(axis=0),
            args.outdir / "role_bias_shift_baseline_vs_self_note.pdf",
        )

    write_report(
        args.outdir / "analysis_report.md",
        baseline_examples,
        [("belief_vector+social", belief_social_examples)]
        + ([("llm_self_note", self_note_examples)] if args.self_note_exp else []),
        three_class,
    )

    summary = {
        "baseline_n": len(baseline_examples),
        "belief_social_n": len(belief_social_examples),
        "three_class_metrics": three_class,
        "output_dir": str(args.outdir),
    }
    if args.self_note_exp:
        summary["self_note_n"] = len(self_note_examples)
    (args.outdir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved analysis to: {args.outdir}")
    print("Generated files:")
    for p in sorted(args.outdir.glob("*.pdf")):
        print(f"  - {p}")
    print(f"  - {args.outdir / 'analysis_report.md'}")
    print(f"  - {args.outdir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
