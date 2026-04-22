#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import plotly.graph_objects as go


ROLE_ORDER = ["assassin", "merlin", "morgana", "percival", "servant-1", "servant-2"]
EVIL_ROLES: Set[str] = {"assassin", "morgana"}
GOOD_ROLES: Set[str] = {"percival", "servant-1", "servant-2"}
MERLIN_ROLE = "merlin"

PAPER_FONT = "Times New Roman"
PAPER_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
]


def _paper_layout(title: str, width: int, height: int, left: int = 110, right: int = 35, top: int = 58, bottom: int = 58) -> Dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "template": "plotly_white",
        "width": width,
        "height": height,
        "margin": {"l": left, "r": right, "t": top, "b": bottom},
        "font": {"family": PAPER_FONT, "size": 12},
    }


def _color_for_label(label: str, ordered_labels: List[str]) -> str:
    if not ordered_labels:
        return PAPER_COLORS[0]
    idx = max(0, ordered_labels.index(label)) if label in ordered_labels else 0
    return PAPER_COLORS[idx % len(PAPER_COLORS)]


@dataclass
class Example:
    true_role: str | None
    pred_role: str | None
    game_id: str | None
    quest: int | None
    llm_processed: Dict[str, Any]
    prompt_chars: int | None


@dataclass
class ConditionBundle:
    label: str
    slug: str
    technique: str
    exp_dir: Path
    mode: str
    examples: List[Example]
    runs: List[List[Example]]


_TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})")


def _slugify(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower())
    s2 = s2.strip("_")
    return s2 or "condition"


def _folder_timestamp_prefix(name: str) -> str:
    m = _TS_RE.match(name)
    return m.group(1) if m else "unknown-time"


def _parse_csv_items(text: str | None) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _infer_roles_from_bundles(bundles: List[ConditionBundle]) -> List[str]:
    seen: Dict[str, None] = {}
    for b in bundles:
        for ex in b.examples:
            if isinstance(ex.true_role, str) and ex.true_role:
                seen.setdefault(ex.true_role, None)
            if isinstance(ex.pred_role, str) and ex.pred_role:
                seen.setdefault(ex.pred_role, None)
    return sorted(seen.keys())


def _set_role_globals(role_order: List[str], evil_roles: Set[str], good_roles: Set[str], merlin_role: str) -> None:
    global ROLE_ORDER, EVIL_ROLES, GOOD_ROLES, MERLIN_ROLE
    ROLE_ORDER = list(role_order)
    EVIL_ROLES = set(evil_roles)
    GOOD_ROLES = set(good_roles)
    MERLIN_ROLE = str(merlin_role)


def _resolve_role_config(
    bundles: List[ConditionBundle],
    role_order_arg: str | None,
    evil_roles_arg: str | None,
    good_roles_arg: str | None,
    merlin_role_arg: str,
    safe_role_arg: str | None,
    focus_role_arg: str | None,
) -> Dict[str, Any]:
    inferred_roles = _infer_roles_from_bundles(bundles)
    grouped_triplet = {"good", "evil", "merlin"}

    role_order_cli = _parse_csv_items(role_order_arg)
    if role_order_cli:
        role_order = list(dict.fromkeys(role_order_cli + [r for r in inferred_roles if r not in role_order_cli]))
    elif inferred_roles:
        role_order = inferred_roles
    else:
        role_order = list(ROLE_ORDER)

    evil_roles_cli = set(_parse_csv_items(evil_roles_arg))
    if evil_roles_cli:
        evil_roles = {r for r in evil_roles_cli if r in role_order}
    elif grouped_triplet.issubset(set(role_order)):
        evil_roles = {"evil"}
    else:
        evil_roles = {r for r in EVIL_ROLES if r in role_order}

    merlin_role = str(merlin_role_arg).strip() if merlin_role_arg else MERLIN_ROLE
    if merlin_role and merlin_role not in role_order:
        merlin_role = ""

    good_roles_cli = set(_parse_csv_items(good_roles_arg))
    if good_roles_cli:
        good_roles = {r for r in good_roles_cli if r in role_order}
    elif grouped_triplet.issubset(set(role_order)):
        good_roles = {"good"}
    else:
        good_roles = {r for r in role_order if r not in evil_roles and r != merlin_role}

    baseline_bundle = _latest_by_technique(bundles, "baseline_full_transcript")
    baseline_examples = baseline_bundle.examples if baseline_bundle is not None else []

    if safe_role_arg and safe_role_arg in role_order:
        safe_role = str(safe_role_arg)
    elif "servant-1" in role_order:
        safe_role = "servant-1"
    elif "servant" in role_order:
        safe_role = "servant"
    else:
        pred_counts = Counter(ex.pred_role for ex in baseline_examples if ex.pred_role in role_order)
        if good_roles:
            good_pred = [(r, pred_counts.get(r, 0)) for r in good_roles]
            safe_role = max(good_pred, key=lambda x: x[1])[0]
        else:
            safe_role = role_order[0]

    if focus_role_arg and focus_role_arg in role_order:
        focus_role = str(focus_role_arg)
    elif "assassin" in role_order:
        focus_role = "assassin"
    elif evil_roles:
        focus_role = sorted(list(evil_roles))[0]
    else:
        focus_role = role_order[0]

    _set_role_globals(role_order, evil_roles, good_roles, merlin_role)
    return {
        "role_order": role_order,
        "evil_roles": sorted(list(evil_roles)),
        "good_roles": sorted(list(good_roles)),
        "merlin_role": merlin_role,
        "safe_role": safe_role,
        "focus_role": focus_role,
    }


def _safe_mean_std(arr: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.zeros(width, dtype=float), np.zeros(width, dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    mean = np.nan_to_num(np.array(mean, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(np.array(std, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return mean, std


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _dynamic_height(
    n_items: int,
    per_item_px: int,
    top: int,
    bottom: int,
    min_height: int,
    max_height: int,
) -> int:
    body = max(1, int(n_items)) * int(per_item_px)
    return _clamp(int(top) + int(bottom) + body, int(min_height), int(max_height))


def _dynamic_left_margin(labels: Iterable[str], base: int = 120, per_char: int = 5, cap: int = 360) -> int:
    max_len = max((len(str(x)) for x in labels), default=0)
    return _clamp(base + max_len * per_char, base, cap)


def _apply_name_disambiguation(bundles: List[ConditionBundle]) -> List[ConditionBundle]:
    """Drop timestamps from labels/slugs unless duplicate technique names require disambiguation."""
    by_technique: Dict[str, List[ConditionBundle]] = {}
    for b in bundles:
        by_technique.setdefault(b.technique, []).append(b)

    for technique, group in by_technique.items():
        if len(group) == 1:
            b = group[0]
            b.label = technique
            b.slug = _slugify(technique)
            continue

        for b in sorted(group, key=lambda x: x.exp_dir.name):
            ts = _folder_timestamp_prefix(b.exp_dir.name)
            b.label = f"{technique} | {ts}"
            b.slug = _slugify(f"{technique}_{ts}")

    return bundles


def _discover_conditions(eval_root: Path) -> List[ConditionBundle]:
    bundles: List[ConditionBundle] = []
    for exp_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        cfg_path = exp_dir / "config.json"
        runs_dir = exp_dir / "runs"
        if not cfg_path.exists() or not runs_dir.exists():
            continue
        try:
            cfg = load_json(cfg_path) or {}
        except Exception:
            continue

        run_dirs = [p for p in runs_dir.glob("run_*") if p.is_dir()]
        if not run_dirs:
            continue

        current_prompt = str(cfg.get("current_prompt") or "none")

        has_baseline = any((rd / "results_baseline.json").exists() for rd in run_dirs)
        has_current = any((rd / "results_current.json").exists() for rd in run_dirs)

        # Dedicated baseline run folders often have current_prompt='none'.
        if current_prompt == "none" and has_baseline:
            label = "baseline_full_transcript"
            slug = _slugify("baseline_full_transcript")
            runs = load_runs_grouped(exp_dir, mode="baseline")
            examples = load_all_runs(exp_dir, mode="baseline")
            if runs:
                bundles.append(
                    ConditionBundle(
                        label=label,
                        slug=slug,
                        technique="baseline_full_transcript",
                        exp_dir=exp_dir,
                        mode="baseline",
                        examples=examples,
                        runs=runs,
                    )
                )
            continue

        if has_current:
            label = current_prompt
            slug = _slugify(current_prompt)
            runs = load_runs_grouped(exp_dir, mode="current")
            examples = load_all_runs(exp_dir, mode="current")
            if runs:
                bundles.append(
                    ConditionBundle(
                        label=label,
                        slug=slug,
                        technique=current_prompt,
                        exp_dir=exp_dir,
                        mode="current",
                        examples=examples,
                        runs=runs,
                    )
                )

    return _apply_name_disambiguation(bundles)


def _latest_by_technique(bundles: List[ConditionBundle], technique: str) -> ConditionBundle | None:
    cands = [b for b in bundles if b.technique == technique]
    if not cands:
        return None
    return sorted(cands, key=lambda b: b.exp_dir.name)[-1]


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


def role_bucket_vector(examples: Iterable[Example], role: str) -> np.ndarray:
    by_q = _bucket_by_quest(examples, role)
    return np.array([float(by_q[k]) for k in ["q1", "q2", "q3", "q4+"]], dtype=float)


def role_bucket_share_vector(examples: Iterable[Example], role: str) -> np.ndarray:
    vec = role_bucket_vector(examples, role)
    s = float(vec.sum())
    return (vec / s) if s > 0 else vec


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _pred_role_counts(examples: Iterable[Example]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for ex in examples:
        if ex.pred_role in ROLE_ORDER:
            c[str(ex.pred_role)] += 1
    return {r: int(c.get(r, 0)) for r in ROLE_ORDER}


def _role_share_from_examples(examples: Iterable[Example], role: str) -> float:
    pred = [ex.pred_role for ex in examples if ex.pred_role in ROLE_ORDER]
    if not pred:
        return 0.0
    return safe_div(sum(1 for p in pred if p == role), len(pred))


def bootstrap_role_share_ci(
    runs: List[List[Example]],
    role: str,
    n_boot: int = 3000,
    seed: int = 17,
) -> Dict[str, float]:
    """Run-aware bootstrap CI for predicted-role share.

    Resamples runs with replacement and then resamples examples inside each
    selected run with replacement.
    """
    if not runs:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    run_sizes = [len(r) for r in runs]
    valid_run_idx = [i for i, n in enumerate(run_sizes) if n > 0]
    if not valid_run_idx:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    vals: List[float] = []
    n_runs = len(valid_run_idx)
    for _ in range(int(n_boot)):
        sampled_run_idx = rng.choice(valid_run_idx, size=n_runs, replace=True)
        total = 0
        hit = 0
        for idx in sampled_run_idx:
            run = runs[int(idx)]
            k = len(run)
            sampled_ex_idx = rng.integers(0, k, size=k)
            for j in sampled_ex_idx:
                pr = run[int(j)].pred_role
                if pr in ROLE_ORDER:
                    total += 1
                    if pr == role:
                        hit += 1
        vals.append(safe_div(hit, total))

    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "ci_low": float(np.quantile(arr, 0.025)) if arr.size else 0.0,
        "ci_high": float(np.quantile(arr, 0.975)) if arr.size else 0.0,
    }


def _chi_square_stat_from_two_rows(row_a: np.ndarray, row_b: np.ndarray) -> Tuple[float, int, int]:
    observed = np.vstack([row_a, row_b]).astype(float)
    total = float(observed.sum())
    if total <= 0.0:
        return 0.0, observed.shape[1] - 1, 0

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(expected > 0.0, np.square(observed - expected) / expected, 0.0)
    stat = float(contrib.sum())
    dof = int(observed.shape[1] - 1)
    return stat, dof, int(total)


def _cramers_v(stat: float, n: int, n_cols: int) -> float:
    if n <= 0:
        return 0.0
    # For 2 x k table, min(r-1, c-1) = 1.
    denom = float(n)
    return float(np.sqrt(max(0.0, stat / denom))) if denom > 0.0 else 0.0


def permutation_test_role_distribution_shift(
    baseline_examples: List[Example],
    target_examples: List[Example],
    n_perm: int = 5000,
    seed: int = 29,
) -> Dict[str, Any]:
    """Permutation test on 2 x K predicted-role contingency table."""
    base_pred = [ex.pred_role for ex in baseline_examples if ex.pred_role in ROLE_ORDER]
    targ_pred = [ex.pred_role for ex in target_examples if ex.pred_role in ROLE_ORDER]
    n_a = len(base_pred)
    n_b = len(targ_pred)
    if n_a == 0 or n_b == 0:
        return {
            "chi_square": 0.0,
            "dof": len(ROLE_ORDER) - 1,
            "p_value": 1.0,
            "cramers_v": 0.0,
            "n_baseline": n_a,
            "n_target": n_b,
        }

    base_row = np.array([sum(1 for r in base_pred if r == role) for role in ROLE_ORDER], dtype=float)
    targ_row = np.array([sum(1 for r in targ_pred if r == role) for role in ROLE_ORDER], dtype=float)
    observed_stat, dof, total_n = _chi_square_stat_from_two_rows(base_row, targ_row)

    rng = np.random.default_rng(seed)
    all_labels = np.array(base_pred + targ_pred, dtype=object)
    ge = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(all_labels)
        pa = perm[:n_a]
        pb = perm[n_a:]
        row_a = np.array([np.sum(pa == role) for role in ROLE_ORDER], dtype=float)
        row_b = np.array([np.sum(pb == role) for role in ROLE_ORDER], dtype=float)
        st, _, _ = _chi_square_stat_from_two_rows(row_a, row_b)
        if st >= observed_stat:
            ge += 1

    p_val = float((ge + 1) / (int(n_perm) + 1))
    return {
        "chi_square": observed_stat,
        "dof": dof,
        "p_value": p_val,
        "cramers_v": _cramers_v(observed_stat, total_n, len(ROLE_ORDER)),
        "n_baseline": n_a,
        "n_target": n_b,
    }


def permutation_test_single_role_share_shift(
    baseline_examples: List[Example],
    target_examples: List[Example],
    role: str,
    n_perm: int = 10000,
    seed: int = 31,
) -> Dict[str, float]:
    base_pred = [ex.pred_role for ex in baseline_examples if ex.pred_role in ROLE_ORDER]
    targ_pred = [ex.pred_role for ex in target_examples if ex.pred_role in ROLE_ORDER]
    n_a = len(base_pred)
    n_b = len(targ_pred)
    if n_a == 0 or n_b == 0:
        return {"delta": 0.0, "p_value": 1.0}

    obs = safe_div(sum(1 for x in targ_pred if x == role), n_b) - safe_div(sum(1 for x in base_pred if x == role), n_a)

    all_labels = np.array(base_pred + targ_pred, dtype=object)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(all_labels)
        pa = perm[:n_a]
        pb = perm[n_a:]
        d = safe_div(np.sum(pb == role), n_b) - safe_div(np.sum(pa == role), n_a)
        if abs(float(d)) >= abs(float(obs)):
            ge += 1

    return {
        "delta": float(obs),
        "p_value": float((ge + 1) / (int(n_perm) + 1)),
    }


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
    if role in {"good", "evil", "merlin"}:
        return role
    if MERLIN_ROLE and role == MERLIN_ROLE:
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
        **_paper_layout(title=title, width=980, height=420, left=130, top=62, bottom=62),
        barmode="group",
        xaxis_title="Role share (mean across runs)",
        yaxis_title="Role",
        legend={
            "font": {"family": PAPER_FONT, "size": 11},
            "orientation": "h",
            "y": 1.08,
            "x": 0.0,
            "traceorder": "reversed",
        },
        xaxis={"range": [0.0, 1.0]},
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_confusion_heatmap(conf: Dict[str, Dict[str, int]], out_pdf: Path, title: str) -> None:
    m = np.array([[conf[tr][pr] for pr in ROLE_ORDER] for tr in ROLE_ORDER], dtype=float)
    row_sums = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, np.where(row_sums == 0, 1.0, row_sums))

    # Keep in-cell text compact for one-column readability.
    text = [[f"{norm[i, j] * 100:.1f}%" for j in range(len(ROLE_ORDER))] for i in range(len(ROLE_ORDER))]
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=norm,
                x=ROLE_ORDER,
                y=ROLE_ORDER,
                colorscale=[[0.0, "#F7FBFF"], [0.5, "#C6DBEF"], [1.0, "#6BAED6"]],
                zmin=0.0,
                zmax=1.0,
                colorbar={"title": "Row-normalized proportion"},
                text=text,
                texttemplate="%{text}",
                textfont={"size": 12, "color": "black", "family": PAPER_FONT},
                hovertemplate="True=%{y}<br>Pred=%{x}<br>Rate=%{z:.3f}<br>Count=%{customdata}<extra></extra>",
                customdata=m.astype(int),
            )
        ]
    )
    fig.update_layout(
        **_paper_layout(title=title, width=1120, height=760, left=135, right=55, top=72, bottom=95),
        xaxis_title="Predicted role",
        yaxis_title="True role",
    )
    fig.update_xaxes(tickangle=20, tickfont={"size": 12})
    fig.update_yaxes(tickfont={"size": 12})
    fig.update_yaxes(autorange="reversed")
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def plot_three_class_f1(
    metrics_by_condition: Dict[str, Dict[str, Any]],
    per_run_three_class: Dict[str, List[Dict[str, Any]]],
    out_pdf: Path,
) -> None:
    classes = ["good", "evil", "merlin"]
    conds = list(metrics_by_condition.keys())
    legend_cols = 4
    legend_rows = max(1, int(math.ceil(len(conds) / legend_cols)))
    top_margin = 62 + max(0, legend_rows - 1) * 20
    fig_height = _dynamic_height(
        n_items=legend_rows,
        per_item_px=46,
        top=top_margin,
        bottom=62,
        min_height=430,
        max_height=840,
    )

    fig = go.Figure()
    cond_colors = {c: _color_for_label(c, conds) for c in conds}
    for cond in conds:
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
                marker_color=cond_colors[cond],
                error_x={"type": "data", "array": errs, "visible": True},
            )
        )
    fig.update_layout(
        **_paper_layout(
            title=f"3-Class F1 (good / evil / {MERLIN_ROLE or 'merlin'})",
            width=980,
            height=fig_height,
            left=125,
            right=40,
            top=top_margin,
            bottom=62,
        ),
        barmode="group",
        xaxis_title="F1",
        yaxis_title="Grouped Class",
        xaxis={"range": [0.0, 1.0]},
        legend={
            "font": {"family": PAPER_FONT, "size": 11},
            "orientation": "h",
            "y": 1.08,
            "x": 0.0,
            "xanchor": "left",
            "traceorder": "reversed",
        },
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


def plot_role_diagnostics(means: np.ndarray, stds: np.ndarray, out_pdf: Path, role_name: str) -> None:
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
        **_paper_layout(title=f"Belief+Social: {role_name} Predictions by Quest Bucket", width=880, height=360, left=120, right=35, top=58, bottom=56),
        xaxis_title=f"Share of predicted {role_name} by quest bucket",
        yaxis_title="Quest Bucket",
        xaxis={"range": [0.0, 1.0]},
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

    left_margin = _dynamic_left_margin(labels, base=165, per_char=5, cap=380)
    fig_height = _dynamic_height(
        n_items=len(labels),
        per_item_px=44,
        top=62,
        bottom=62,
        min_height=430,
        max_height=1300,
    )

    fig = go.Figure()
    cond_colors = {c: _color_for_label(c, labels) for c in labels}
    for label, vals in zip(labels, data):
        fig.add_trace(
            go.Box(
                x=vals,
                y=[label] * len(vals),
                orientation="h",
                name=label,
                marker_color=cond_colors[label],
                boxpoints=False,
                showlegend=False,
            )
        )
    fig.update_layout(
        **_paper_layout(
            title="Run-Level 3-Class Micro-F1 Distribution",
            width=980,
            height=fig_height,
            left=left_margin,
            right=35,
            top=62,
            bottom=62,
        ),
        xaxis_title="3-class micro-F1",
        yaxis_title="Condition",
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
        **_paper_layout(title="Prediction-Bias Shift by Role", width=980, height=420, left=130, right=35, top=62, bottom=62),
        xaxis_title="Predicted role share shift (belief+social - baseline)",
        yaxis_title="Role",
    )
    fig.write_image(str(out_pdf), format="pdf", scale=2)


def write_report(
    out_path: Path,
    baseline_examples: List[Example],
    focus_sections: List[Tuple[str, List[Example]]],
    three_class: Dict[str, Dict[str, Any]],
    safe_role: str,
    focus_role: str,
    hypothesis_tests: Dict[str, Any] | None = None,
) -> None:
    base_pred = Counter(ex.pred_role for ex in baseline_examples if ex.pred_role)

    base_true_break = _true_role_breakdown_for_pred(baseline_examples, safe_role)

    lines: List[str] = []
    lines.append("# Explainability Analysis Report")
    lines.append("")
    lines.append(f"## 1) Why baseline focuses on {safe_role}")
    lines.append(f"- Share of {safe_role} predictions: {(_top_share(base_pred, safe_role)*100):.2f}%")
    lines.append(f"- True-role composition among predictions labeled {safe_role}:")
    for r in ROLE_ORDER:
        lines.append(f"  - {r}: {base_true_break.get(r, 0)}")
    lines.append("- Interpretation:")
    lines.append("  - Baseline tends to collapse toward a majority safe class when evidence is ambiguous.")
    lines.append("  - This is consistent with high confusion into servant classes in aggregate confusion matrices.")
    lines.append("")
    lines.append("## 2) Focus-technique behavior")
    for focus_label, focus_examples in focus_sections:
        focus_pred = Counter(ex.pred_role for ex in focus_examples if ex.pred_role)
        focus_top_role = max(focus_pred.items(), key=lambda x: x[1])[0] if focus_pred else focus_role
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
    if hypothesis_tests:
        lines.append("")
        lines.append("## 4) Hypothesis tests and uncertainty")
        bs = hypothesis_tests.get("baseline_vs_belief_social", {}) or {}
        s1 = bs.get("safe_role_share", {}) or {}
        ass = bs.get("focus_role_share", {}) or {}
        dist = bs.get("distribution_shift_test", {}) or {}
        lines.append("### baseline vs belief+social")
        lines.append(
            f"- {safe_role} share: "
            f"baseline={float(s1.get('baseline_share', 0.0)):.4f} "
            f"(95% CI [{float(s1.get('baseline_ci_low', 0.0)):.4f}, {float(s1.get('baseline_ci_high', 0.0)):.4f}]), "
            f"belief+social={float(s1.get('target_share', 0.0)):.4f} "
            f"(95% CI [{float(s1.get('target_ci_low', 0.0)):.4f}, {float(s1.get('target_ci_high', 0.0)):.4f}]), "
            f"delta={float(s1.get('delta', 0.0)):.4f}, permutation p={float(s1.get('perm_p_value', 1.0)):.6f}"
        )
        lines.append(
            f"- {focus_role} share: "
            f"baseline={float(ass.get('baseline_share', 0.0)):.4f} "
            f"(95% CI [{float(ass.get('baseline_ci_low', 0.0)):.4f}, {float(ass.get('baseline_ci_high', 0.0)):.4f}]), "
            f"belief+social={float(ass.get('target_share', 0.0)):.4f} "
            f"(95% CI [{float(ass.get('target_ci_low', 0.0)):.4f}, {float(ass.get('target_ci_high', 0.0)):.4f}]), "
            f"delta={float(ass.get('delta', 0.0)):.4f}, permutation p={float(ass.get('perm_p_value', 1.0)):.6f}"
        )
        lines.append(
            f"- full predicted-role distribution shift (2x{len(ROLE_ORDER)}): "
            f"chi-square={float(dist.get('chi_square', 0.0)):.4f}, dof={int(dist.get('dof', 0))}, "
            f"permutation p={float(dist.get('p_value', 1.0)):.6f}, Cramer's V={float(dist.get('cramers_v', 0.0)):.4f}"
        )
        lines.append("- Interpretation discipline:")
        lines.append("  - Significant p-values support inferential claims about distribution shifts.")
        lines.append("  - Mechanistic claims still rely on correlated diagnostics (repair/override/decision path), not causal identification.")
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
    focus_role: str,
) -> None:
    if not cond_runs:
        print(f"[warn] Skipping '{cond_label}' because no runs were loaded.")
        return

    true_arr = np.array([role_share_vector(run, source="true") for run in cond_runs], dtype=float)
    pred_arr = np.array([role_share_vector(run, source="pred") for run in cond_runs], dtype=float)
    true_mean, true_std = _safe_mean_std(true_arr, len(ROLE_ORDER))
    pred_mean, pred_std = _safe_mean_std(pred_arr, len(ROLE_ORDER))

    plot_distribution_comparison(
        true_mean,
        true_std,
        pred_mean,
        pred_std,
        outdir / f"{cond_slug}_true_vs_pred_distribution.pdf",
        f"{cond_label}: True vs Predicted Role Distribution",
    )

    conf = count_confusion(cond_examples)
    plot_confusion_heatmap(
        conf,
        outdir / f"{cond_slug}_confusion_heatmap.pdf",
        f"{cond_label} Confusion Matrix (Row-normalized)",
    )

    focus_bucket_arr = np.array([role_bucket_share_vector(run, focus_role) for run in cond_runs], dtype=float)
    ass_mean, ass_std = _safe_mean_std(focus_bucket_arr, 4)
    plot_role_diagnostics(
        ass_mean,
        ass_std,
        outdir / f"{cond_slug}_{_slugify(focus_role)}_by_quest_bucket.pdf",
        role_name=focus_role,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="In-depth explainability analysis for Avalon role-inference experiments.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Optional root directory containing many experiment folders. When set, experiments are auto-discovered via each folder's config.json current_prompt.",
    )
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
    parser.add_argument("--role_order", type=str, default=None, help="Optional comma-separated role order for plots/matrices.")
    parser.add_argument("--evil_roles", type=str, default=None, help="Optional comma-separated evil roles.")
    parser.add_argument("--good_roles", type=str, default=None, help="Optional comma-separated good roles.")
    parser.add_argument("--merlin_role", type=str, default="merlin", help="Role name treated as merlin class for grouped metrics.")
    parser.add_argument("--safe_role", type=str, default=None, help="Optional baseline focus role in report/hypothesis (e.g., servant or servant-1).")
    parser.add_argument("--focus_role", type=str, default=None, help="Optional role used for quest-bucket diagnostics (e.g., assassin).")
    args = parser.parse_args()

    _configure_plot_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.eval_root:
        bundles = _discover_conditions(args.eval_root)
        if not bundles:
            raise SystemExit(f"No experiment bundles discovered under: {args.eval_root}")
        print(f"[info] Discovered {len(bundles)} condition bundles from {args.eval_root}")
    else:
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

        bundles = [
            ConditionBundle("baseline", "baseline", "baseline_full_transcript", args.baseline_exp, "baseline", baseline_examples, baseline_runs),
            ConditionBundle("vector_memory", "vector_memory", "vector_memory", args.vector_exp, "current", vector_examples, vector_runs),
            ConditionBundle("belief_vector", "belief_vector", "belief_vector", args.belief_exp, "current", belief_examples, belief_runs),
            ConditionBundle("belief_vector+social", "belief_social", "belief_vector+social", args.belief_social_exp, "current", belief_social_examples, belief_social_runs),
        ]
        if args.self_note_exp:
            bundles.append(ConditionBundle("llm_self_note", "self_note", "llm_self_note", args.self_note_exp, "current", self_note_examples, self_note_runs))

    bundles = [b for b in bundles if b.runs]
    if not bundles:
        raise SystemExit("No non-empty condition bundles found.")

    role_cfg = _resolve_role_config(
        bundles=bundles,
        role_order_arg=args.role_order,
        evil_roles_arg=args.evil_roles,
        good_roles_arg=args.good_roles,
        merlin_role_arg=args.merlin_role,
        safe_role_arg=args.safe_role,
        focus_role_arg=args.focus_role,
    )
    print(
        f"[info] Role config: roles={role_cfg['role_order']} | evil={role_cfg['evil_roles']} | "
        f"good={role_cfg['good_roles']} | merlin={role_cfg['merlin_role']} | "
        f"safe_role={role_cfg['safe_role']} | focus_role={role_cfg['focus_role']}"
    )

    baseline_bundle = _latest_by_technique(bundles, "baseline_full_transcript")
    belief_social_bundle = _latest_by_technique(bundles, "belief_vector+social")

    if baseline_bundle is not None:
        base_true_arr = np.array([role_share_vector(run, source="true") for run in baseline_bundle.runs], dtype=float)
        base_pred_arr = np.array([role_share_vector(run, source="pred") for run in baseline_bundle.runs], dtype=float)
        base_true_mean, base_true_std = _safe_mean_std(base_true_arr, len(ROLE_ORDER))
        base_pred_mean, base_pred_std = _safe_mean_std(base_pred_arr, len(ROLE_ORDER))
        plot_distribution_comparison(
            base_true_mean,
            base_true_std,
            base_pred_mean,
            base_pred_std,
            args.outdir / "baseline_true_vs_pred_distribution.pdf",
            "Baseline: True vs Predicted Role Distribution",
        )
        baseline_conf = count_confusion(baseline_bundle.examples)
        plot_confusion_heatmap(
            baseline_conf,
            args.outdir / "baseline_confusion_heatmap.pdf",
            "Baseline Confusion Matrix (Row-normalized)",
        )

    for b in bundles:
        if b.technique == "baseline_full_transcript":
            continue
        _plot_condition_bundle(b.label, b.slug, b.examples, b.runs, args.outdir, focus_role=str(role_cfg["focus_role"]))

    conditions: List[Tuple[str, List[Example], List[List[Example]]]] = [(b.label, b.examples, b.runs) for b in bundles]
    three_class = {name: three_class_metrics(exs) for name, exs, _ in conditions}
    per_run_three_class = {name: [three_class_metrics(run) for run in runs] for name, _, runs in conditions}
    plot_three_class_f1(three_class, per_run_three_class, args.outdir / "three_class_f1_comparison.pdf")
    plot_micro_f1_boxplot(per_run_three_class, args.outdir / "three_class_micro_f1_boxplot.pdf")

    if baseline_bundle is not None:
        base_share_arr = np.array([role_share_vector(run, source="pred") for run in baseline_bundle.runs], dtype=float)
        base_share_mean, base_share_std = _safe_mean_std(base_share_arr, len(ROLE_ORDER))
        for b in bundles:
            if b.technique == "baseline_full_transcript":
                continue
            t_arr = np.array([role_share_vector(run, source="pred") for run in b.runs], dtype=float)
            t_mean, t_std = _safe_mean_std(t_arr, len(ROLE_ORDER))
            plot_role_bias_shift(
                base_share_mean,
                t_mean,
                base_share_std,
                t_std,
                args.outdir / f"role_bias_shift_baseline_vs_{b.slug}.pdf",
            )

    hypothesis_tests: Dict[str, Any] = {}
    if baseline_bundle is not None and belief_social_bundle is not None:
        safe_role = str(role_cfg["safe_role"])
        focus_role = str(role_cfg["focus_role"])
        s1_base_ci = bootstrap_role_share_ci(baseline_bundle.runs, role=safe_role, n_boot=3000, seed=101)
        s1_bs_ci = bootstrap_role_share_ci(belief_social_bundle.runs, role=safe_role, n_boot=3000, seed=102)
        ass_base_ci = bootstrap_role_share_ci(baseline_bundle.runs, role=focus_role, n_boot=3000, seed=103)
        ass_bs_ci = bootstrap_role_share_ci(belief_social_bundle.runs, role=focus_role, n_boot=3000, seed=104)

        s1_perm = permutation_test_single_role_share_shift(
            baseline_bundle.examples,
            belief_social_bundle.examples,
            role=safe_role,
            n_perm=10000,
            seed=201,
        )
        ass_perm = permutation_test_single_role_share_shift(
            baseline_bundle.examples,
            belief_social_bundle.examples,
            role=focus_role,
            n_perm=10000,
            seed=202,
        )
        dist_perm = permutation_test_role_distribution_shift(
            baseline_bundle.examples,
            belief_social_bundle.examples,
            n_perm=5000,
            seed=203,
        )

        baseline_pred_share = role_share_vector(baseline_bundle.examples, source="pred")
        belief_social_pred_share = role_share_vector(belief_social_bundle.examples, source="pred")

        hypothesis_tests = {
            "baseline_vs_belief_social": {
                "safe_role": safe_role,
                "focus_role": focus_role,
                "safe_role_share": {
                    "baseline_share": float(baseline_pred_share[ROLE_ORDER.index(safe_role)]),
                    "baseline_ci_low": float(s1_base_ci["ci_low"]),
                    "baseline_ci_high": float(s1_base_ci["ci_high"]),
                    "target_share": float(belief_social_pred_share[ROLE_ORDER.index(safe_role)]),
                    "target_ci_low": float(s1_bs_ci["ci_low"]),
                    "target_ci_high": float(s1_bs_ci["ci_high"]),
                    "delta": float(s1_perm["delta"]),
                    "perm_p_value": float(s1_perm["p_value"]),
                },
                "focus_role_share": {
                    "baseline_share": float(baseline_pred_share[ROLE_ORDER.index(focus_role)]),
                    "baseline_ci_low": float(ass_base_ci["ci_low"]),
                    "baseline_ci_high": float(ass_base_ci["ci_high"]),
                    "target_share": float(belief_social_pred_share[ROLE_ORDER.index(focus_role)]),
                    "target_ci_low": float(ass_bs_ci["ci_low"]),
                    "target_ci_high": float(ass_bs_ci["ci_high"]),
                    "delta": float(ass_perm["delta"]),
                    "perm_p_value": float(ass_perm["p_value"]),
                },
                "distribution_shift_test": dist_perm,
                "n_baseline_examples": len(baseline_bundle.examples),
                "n_belief_social_examples": len(belief_social_bundle.examples),
                "baseline_bundle": baseline_bundle.label,
                "belief_social_bundle": belief_social_bundle.label,
            }
        }

    write_report(
        args.outdir / "analysis_report.md",
        baseline_bundle.examples if baseline_bundle is not None else [],
        [(b.label, b.examples) for b in bundles if b.technique != "baseline_full_transcript"],
        three_class,
        safe_role=str(role_cfg["safe_role"]),
        focus_role=str(role_cfg["focus_role"]),
        hypothesis_tests=hypothesis_tests or None,
    )

    summary = {
        "baseline_n": len(baseline_bundle.examples) if baseline_bundle is not None else 0,
        "belief_social_n": len(belief_social_bundle.examples) if belief_social_bundle is not None else 0,
        "three_class_metrics": three_class,
        "role_config": role_cfg,
        "hypothesis_tests": hypothesis_tests,
        "output_dir": str(args.outdir),
        "bundles": [
            {
                "label": b.label,
                "slug": b.slug,
                "technique": b.technique,
                "exp_dir": str(b.exp_dir),
                "mode": b.mode,
                "n_examples": len(b.examples),
                "n_runs": len(b.runs),
            }
            for b in bundles
        ],
    }
    (args.outdir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.outdir / "hypothesis_tests.json").write_text(json.dumps(hypothesis_tests, indent=2), encoding="utf-8")

    print(f"Saved analysis to: {args.outdir}")
    print("Generated files:")
    for p in sorted(args.outdir.glob("*.pdf")):
        print(f"  - {p}")
    print(f"  - {args.outdir / 'analysis_report.md'}")
    print(f"  - {args.outdir / 'summary_metrics.json'}")
    print(f"  - {args.outdir / 'hypothesis_tests.json'}")


if __name__ == "__main__":
    main()
