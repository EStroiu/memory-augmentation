#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.offline import plot as plotly_offline_plot


def avg_numbers(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def merge_avg_prompt_sizes(items: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    out = {
        "with_memory": {"chars": 0.0, "words": 0.0, "lines": 0.0},
        "baseline": {"chars": 0.0, "words": 0.0, "lines": 0.0},
    }
    if not items:
        return out
    for key1 in out.keys():
        for key2 in out[key1].keys():
            out[key1][key2] = float(np.mean([x.get(key1, {}).get(key2, 0.0) for x in items]))
    return out


def average_aggregates(aggs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not aggs:
        return {}
    avg_prompt_sizes = merge_avg_prompt_sizes([a.get("avg_prompt_sizes", {}) for a in aggs])
    roles_clf = set()
    for a in aggs:
        roles_clf |= set(a.get("clf_by_role", {}).keys())
    clf_by_role: Dict[str, Dict[str, float]] = {}
    for r in sorted(roles_clf):
        vals = [a.get("clf_by_role", {}).get(r, {}) for a in aggs]
        clf_by_role[r] = {
            "precision": avg_numbers([v.get("precision", 0.0) for v in vals]),
            "recall": avg_numbers([v.get("recall", 0.0) for v in vals]),
            "f1": avg_numbers([v.get("f1", 0.0) for v in vals]),
        }
    micro_f1 = avg_numbers([a.get("clf_micro_f1", 0.0) for a in aggs])

    conf_roles = sorted(roles_clf)
    conf: Dict[str, Dict[str, float]] = {tr: {pr: 0.0 for pr in conf_roles} for tr in conf_roles}
    for a in aggs:
        cm = a.get("clf_confusion", {}) or {}
        for tr in conf_roles:
            row = cm.get(tr, {}) or {}
            for pr in conf_roles:
                conf[tr][pr] += float(row.get(pr, 0))

    base = aggs[0]
    return {
        "k": base.get("k"),
        "memory_format": base.get("memory_format"),
        "model": base.get("model"),
        "avg_prompt_sizes": avg_prompt_sizes,
        "clf_by_role": clf_by_role,
        "clf_micro_f1": micro_f1,
        "clf_confusion": conf,
    }


def viz_prompt_lengths(run_dir, avg_stats: Dict[str, Dict[str, float]]) -> None:
    if go is None or plotly_offline_plot is None:
        return
    metrics = ["chars", "words", "lines"]
    wm = [avg_stats["with_memory"].get(m, 0.0) for m in metrics]
    bl = [avg_stats["baseline"].get(m, 0.0) for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="with_memory", x=metrics, y=wm))
    fig.add_trace(go.Bar(name="baseline", x=metrics, y=bl))
    fig.update_layout(barmode="group", title="Average prompt sizes", yaxis_title="Count")
    plotly_offline_plot(fig, filename=str(run_dir / "prompt_lengths.html"), auto_open=False, include_plotlyjs="cdn")


def viz_confusion_matrix(
    run_dir,
    title: str,
    roles: List[str],
    confusion: Dict[str, Dict[str, float]],
    f1_by_role: Dict[str, float] | None,
    filename: str,
) -> None:
    if go is None or plotly_offline_plot is None:
        return
    n = len(roles)
    M = np.zeros((n, n), dtype=float)
    for i, tr in enumerate(roles):
        row = confusion.get(tr, {}) or {}
        for j, pr in enumerate(roles):
            M[i, j] = float(row.get(pr, 0.0))
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, np.where(row_sums == 0, 1.0, row_sums), where=np.ones_like(M, dtype=bool))
    text: List[List[str]] = []
    for i in range(n):
        r: List[str] = []
        for j in range(n):
            pct = norm[i, j] * 100.0
            cell = f"{pct:.1f}%\n({int(M[i, j])})"
            if i == j and f1_by_role is not None:
                role = roles[i]
                f1 = float(f1_by_role.get(role, 0.0)) if role in f1_by_role else 0.0
                cell = f"{pct:.1f}%\n({int(M[i, j])})\nF1={f1:.2f}"
            r.append(cell)
        text.append(r)
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
        colorbar=dict(title="Row-normalized"),
    ))
    fig.update_layout(title=title, xaxis_title="Predicted role", yaxis_title="True role")
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")
