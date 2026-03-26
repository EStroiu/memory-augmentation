#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot

EVIL_ROLES = {"assassin", "morgana"}

PARTY_RE = re.compile(r"proposed a party:\s*(.*)$")
YES_NO_RE = re.compile(r"([^:,]+):\s*(yes|no)", flags=re.IGNORECASE)

ACCUSATION_PATTERNS = [
    re.compile(r"\b(?:sus|suspicious|evil|liar|lying|bad|untrustworthy|doubt)\b", flags=re.IGNORECASE),
]
DEFENSE_PATTERNS = [
    re.compile(r"\b(?:good|trust|innocent|safe|clear|defend)\b", flags=re.IGNORECASE),
]

TARGET_REF_RE = re.compile(r"\bplayer-\d+\b", flags=re.IGNORECASE)


@dataclass
class PlayerState:
    p_evil: float = 0.5
    talk: float = 0.0
    yes_total: int = 0
    yes_on_failed: int = 0
    yes_on_success: int = 0
    failed_team_count: int = 0
    success_team_count: int = 0
    contradiction_flips: int = 0
    speaker_turns: int = 0
    accuse_count: int = 0
    defend_count: int = 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = _clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def messages_by_quest(game: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    msgs = list((game.get("messages") or {}).values())
    msgs.sort(key=lambda m: (int(m.get("quest", 0)), int(m.get("turn", 0)), str(m.get("mid", ""))))
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for m in msgs:
        q = int(m.get("quest", 0))
        grouped.setdefault(q, []).append(m)
    return grouped


def parse_party(msg: str) -> List[str]:
    m = PARTY_RE.search(msg or "")
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def parse_vote_outcome(msg: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(msg, str) or "party vote outcome:" not in msg:
        return out
    for player, vote in YES_NO_RE.findall(msg):
        out[str(player).strip()] = str(vote).strip().lower()
    return out


def infer_stance(text: str) -> int:
    tl = (text or "").lower()
    acc = any(p.search(tl) for p in ACCUSATION_PATTERNS)
    deff = any(p.search(tl) for p in DEFENSE_PATTERNS)
    if acc and not deff:
        return -1
    if deff and not acc:
        return 1
    return 0


def referenced_players(text: str, players: List[str]) -> List[str]:
    found = {m.group(0).lower() for m in TARGET_REF_RE.finditer(text or "")}
    pset = {p.lower(): p for p in players}
    return [pset[x] for x in found if x in pset]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - (pos * (pos + 1) / 2.0)
    return float(u / (pos * neg))


def normal_2sided_p_from_t(t_abs: float) -> float:
    return float(math.erfc(float(t_abs) / math.sqrt(2.0)))


def permutation_pvalue_mean_diff(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    yb = y.astype(int)
    x1 = x[yb == 1]
    x0 = x[yb == 0]
    if x1.size == 0 or x0.size == 0:
        return float("nan")
    obs = abs(float(x1.mean() - x0.mean()))
    ge = 0
    for _ in range(int(n_perm)):
        y_perm = rng.permutation(yb)
        p1 = x[y_perm == 1]
        p0 = x[y_perm == 0]
        stat = abs(float(p1.mean() - p0.mean()))
        if stat >= obs:
            ge += 1
    return float((ge + 1) / (int(n_perm) + 1))


def fit_linear_probability_model(df: pd.DataFrame, feature_cols: List[str], target_col: str, seed: int, n_boot: int) -> Dict[str, Any]:
    y = df[target_col].to_numpy(dtype=float)
    X_raw = df[feature_cols].to_numpy(dtype=float)

    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    Xz = (X_raw - means) / stds

    X = np.column_stack([np.ones(len(df), dtype=float), Xz])
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    n = int(X.shape[0])
    p = int(X.shape[1])
    dof = max(1, n - p)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    tvals = beta / se
    pvals = np.array([normal_2sided_p_from_t(abs(t)) for t in tvals], dtype=float)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / max(1, n - p))

    auc = auc_from_scores(y.astype(int), y_hat)

    names = ["intercept"] + feature_cols
    coeff_df = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "std_error": se,
            "t_value": tvals,
            "p_value_normal_approx": pvals,
        }
    )

    rng = np.random.default_rng(seed)
    boot_coefs: List[np.ndarray] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        Xb = X[idx, :]
        yb = y[idx]
        bb = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ yb)
        boot_coefs.append(bb)

    boot_arr = np.vstack(boot_coefs) if boot_coefs else np.zeros((0, len(names)), dtype=float)
    if boot_arr.size > 0:
        low = np.quantile(boot_arr, 0.025, axis=0)
        high = np.quantile(boot_arr, 0.975, axis=0)
        coeff_df["coef_boot_ci_low"] = low
        coeff_df["coef_boot_ci_high"] = high
    else:
        coeff_df["coef_boot_ci_low"] = np.nan
        coeff_df["coef_boot_ci_high"] = np.nan

    return {
        "coefficients": coeff_df,
        "r2": float(r2),
        "adjusted_r2": float(adj_r2),
        "auc_from_linear_score": float(auc) if not math.isnan(auc) else None,
        "n_rows": int(n),
        "n_features": len(feature_cols),
        "feature_standardization": {
            feature_cols[i]: {"mean": float(means[i]), "std": float(stds[i])} for i in range(len(feature_cols))
        },
    }


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    Xz = (X - means) / stds

    out: List[Dict[str, Any]] = []
    for i, feat in enumerate(feature_cols):
        y = Xz[:, i]
        others = [j for j in range(Xz.shape[1]) if j != i]
        if not others:
            out.append({"feature": feat, "vif": 1.0})
            continue
        Xo = Xz[:, others]
        Xo = np.column_stack([np.ones(Xo.shape[0], dtype=float), Xo])
        beta = np.linalg.pinv(Xo.T @ Xo) @ (Xo.T @ y)
        y_hat = Xo @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float(((y - y_hat) ** 2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif = 1.0 / max(1e-9, 1.0 - r2)
        out.append({"feature": feat, "vif": float(vif)})
    return pd.DataFrame(out).sort_values("vif", ascending=False).reset_index(drop=True)


def extract_feature_rows(dataset_dir: Path, talk_alpha: float = 0.3, w_fail: float = 0.8, w_success: float = -0.4) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for path in sorted(dataset_dir.glob("*.json"), key=lambda p: p.name):
        game = load_json(path)
        game_id = path.stem

        users = game.get("users") or {}
        players = sorted([str(u.get("name")) for u in users.values() if u.get("name")], key=lambda x: int(x.split("-")[-1]))
        role_by_player = {
            str(u.get("name")): str(u.get("role"))
            for u in users.values()
            if u.get("name") is not None and u.get("role") is not None
        }

        if not players:
            continue

        states: Dict[str, PlayerState] = {p: PlayerState() for p in players}
        stance_last: Dict[Tuple[str, str], int] = {}

        grouped = messages_by_quest(game)
        for quest in sorted(grouped.keys()):
            msgs = grouped[quest]

            # Feature snapshot before consuming current-quest evidence.
            for player in players:
                st = states[player]
                yes_rate = (st.yes_total / max(1, (quest - 1)))
                contradiction_rate = st.contradiction_flips / max(1, st.speaker_turns)
                rows.append(
                    {
                        "game_id": game_id,
                        "quest": int(quest),
                        "player": player,
                        "true_role": role_by_player.get(player, "unknown"),
                        "is_evil": int(role_by_player.get(player, "") in EVIL_ROLES),
                        "p_evil_prior": float(st.p_evil),
                        "yes_on_failed_prior": int(st.yes_on_failed),
                        "yes_on_success_prior": int(st.yes_on_success),
                        "yes_total_prior": int(st.yes_total),
                        "yes_rate_prior": float(yes_rate),
                        "failed_team_count_prior": int(st.failed_team_count),
                        "success_team_count_prior": int(st.success_team_count),
                        "talk_prior": float(st.talk),
                        "contradiction_flips_prior": int(st.contradiction_flips),
                        "contradiction_rate_prior": float(contradiction_rate),
                        "accuse_count_prior": int(st.accuse_count),
                        "defend_count_prior": int(st.defend_count),
                    }
                )

            party_members: List[str] = []
            vote_map: Dict[str, str] = {}
            quest_failed: bool | None = None
            speaker_counts: Counter[str] = Counter()

            for m in msgs:
                speaker = str(m.get("player", ""))
                text = str(m.get("msg", ""))

                if speaker.lower() == "system":
                    if "proposed a party:" in text:
                        party_members = parse_party(text)
                    if text.startswith("party vote outcome:"):
                        vote_map = parse_vote_outcome(text)
                    if text.endswith("quest failed!"):
                        quest_failed = True
                    elif text.endswith("quest succeeded!"):
                        quest_failed = False
                    continue

                if speaker in states:
                    speaker_counts[speaker] += 1
                    states[speaker].speaker_turns += 1

                stance = infer_stance(text)
                targets = referenced_players(text, players)
                if stance != 0 and targets:
                    for tgt in targets:
                        if tgt == speaker:
                            continue
                        key = (speaker, tgt)
                        prev = stance_last.get(key, 0)
                        if prev != 0 and prev != stance:
                            states[speaker].contradiction_flips += 1
                        stance_last[key] = stance
                        if stance < 0:
                            states[speaker].accuse_count += 1
                        if stance > 0:
                            states[speaker].defend_count += 1

            total_turns = int(sum(speaker_counts.values()))
            if total_turns > 0:
                for p in players:
                    share = float(speaker_counts.get(p, 0)) / float(total_turns)
                    states[p].talk = (1.0 - talk_alpha) * states[p].talk + talk_alpha * share
                    states[p].talk = _clamp(states[p].talk, 0.0, 1.0)

            for p, v in vote_map.items():
                if p not in states:
                    continue
                if v == "yes":
                    states[p].yes_total += 1
                    if quest_failed is True:
                        states[p].yes_on_failed += 1
                    elif quest_failed is False:
                        states[p].yes_on_success += 1

            if party_members and quest_failed is not None:
                delta = w_fail if quest_failed else w_success
                for p in party_members:
                    if p not in states:
                        continue
                    st = states[p]
                    st.p_evil = _clamp(_sigmoid(_logit(st.p_evil) + delta), 0.01, 0.99)
                    if quest_failed:
                        st.failed_team_count += 1
                    else:
                        st.success_team_count += 1

    return pd.DataFrame(rows)


def compute_univariate(df: pd.DataFrame, feature_cols: List[str], target_col: str, n_perm: int, seed: int) -> pd.DataFrame:
    y = df[target_col].to_numpy(dtype=int)
    out: List[Dict[str, Any]] = []

    for i, c in enumerate(feature_cols):
        x = df[c].to_numpy(dtype=float)
        x1 = x[y == 1]
        x0 = x[y == 0]

        m1 = float(x1.mean()) if x1.size else float("nan")
        m0 = float(x0.mean()) if x0.size else float("nan")
        diff = m1 - m0

        v1 = float(x1.var(ddof=1)) if x1.size > 1 else 0.0
        v0 = float(x0.var(ddof=1)) if x0.size > 1 else 0.0
        pooled = math.sqrt(max(1e-12, 0.5 * (v1 + v0)))
        cohen_d = diff / pooled if pooled > 0 else 0.0

        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        auc = auc_from_scores(y, x)
        p_perm = permutation_pvalue_mean_diff(x, y, n_perm=n_perm, seed=seed + i)

        out.append(
            {
                "feature": c,
                "mean_evil": m1,
                "mean_good": m0,
                "mean_diff_evil_minus_good": diff,
                "cohen_d": cohen_d,
                "point_biserial_corr": corr,
                "auc": float(auc) if not math.isnan(auc) else None,
                "p_value_permutation": p_perm,
            }
        )

    udf = pd.DataFrame(out)
    udf["abs_mean_diff"] = udf["mean_diff_evil_minus_good"].abs()
    udf = udf.sort_values(["p_value_permutation", "abs_mean_diff"], ascending=[True, False]).reset_index(drop=True)
    return udf


def save_plot_univariate_auc(udf: pd.DataFrame, out_html: Path) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=udf["feature"],
            y=udf["auc"],
            marker_color="#2E6F95",
        )
    )
    fig.update_layout(
        title="Univariate AUC by Feature (Evil vs Good)",
        xaxis_title="Feature",
        yaxis_title="AUC",
        template="plotly_white",
        yaxis=dict(range=[0.0, 1.0]),
    )
    plotly_offline_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def save_plot_coefficients(coeff_df: pd.DataFrame, out_html: Path) -> None:
    cdf = coeff_df[coeff_df["term"] != "intercept"].copy()
    cdf = cdf.sort_values("coef", ascending=False)

    err_plus = (cdf["coef_boot_ci_high"] - cdf["coef"]).to_numpy(dtype=float)
    err_minus = (cdf["coef"] - cdf["coef_boot_ci_low"]).to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cdf["term"],
            y=cdf["coef"],
            marker_color="#3C8DAD",
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
        )
    )
    fig.update_layout(
        title="Linear Probability Model Coefficients (Standardized Features)",
        xaxis_title="Feature",
        yaxis_title="Coefficient",
        template="plotly_white",
    )
    plotly_offline_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def save_plot_feature_box(df: pd.DataFrame, features: List[str], out_html: Path) -> None:
    fig = go.Figure()
    label = df["is_evil"].map({0: "good", 1: "evil"})
    palette = {"good": "#7FB069", "evil": "#D1495B"}

    for feat in features:
        for cls in ["good", "evil"]:
            vals = df.loc[label == cls, feat]
            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[feat] * len(vals),
                    name=f"{feat} ({cls})",
                    marker_color=palette[cls],
                    boxmean=True,
                    legendgroup=cls,
                    showlegend=(feat == features[0]),
                )
            )

    fig.update_layout(
        title="Feature Distribution by True Class",
        xaxis_title="Feature",
        yaxis_title="Value",
        template="plotly_white",
        boxmode="group",
    )
    plotly_offline_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def write_report_md(out_path: Path, summary: Dict[str, Any], udf: pd.DataFrame, coeff_df: pd.DataFrame) -> None:
    top_u = udf.head(8)
    top_coef = coeff_df[coeff_df["term"] != "intercept"].copy().sort_values("p_value_normal_approx")

    lines: List[str] = []
    lines.append("# Heuristic Feature Audit")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Rows (player-quest snapshots): {summary['n_rows']}")
    lines.append(f"- Games processed: {summary['n_games']}")
    lines.append(f"- Evil ratio: {summary['evil_ratio']:.4f}")
    lines.append("")
    lines.append("## Linear Model Summary")
    lines.append(f"- R^2: {summary['linear_model']['r2']:.4f}")
    lines.append(f"- Adjusted R^2: {summary['linear_model']['adjusted_r2']:.4f}")
    lines.append(f"- AUC from linear score: {summary['linear_model']['auc_from_linear_score']}")
    lines.append(f"- Max VIF: {summary['linear_model']['max_vif']:.3f}")
    lines.append("")
    lines.append("## Top Univariate Signals")
    for _, r in top_u.iterrows():
        lines.append(
            "- "
            + f"{r['feature']}: AUC={r['auc']}, diff={r['mean_diff_evil_minus_good']:.4f}, "
            + f"perm-p={r['p_value_permutation']:.4g}, corr={r['point_biserial_corr']:.4f}"
        )
    lines.append("")
    lines.append("## Most Significant Multivariate Coefficients")
    for _, r in top_coef.head(8).iterrows():
        lines.append(
            "- "
            + f"{r['term']}: coef={r['coef']:.4f}, p~{r['p_value_normal_approx']:.4g}, "
            + f"95% bootstrap CI=[{r['coef_boot_ci_low']:.4f}, {r['coef_boot_ci_high']:.4f}]"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Features with higher AUC and low permutation p-values are stronger standalone signals.")
    lines.append("- Features with stable non-zero standardized coefficients (bootstrap CI away from 0) are more useful jointly.")
    lines.append("- High VIF indicates collinearity and unstable coefficient signs; prioritize low-VIF features.")
    lines.append("- This script evaluates role separability (evil vs good) from handcrafted memory heuristics only.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_base = Path(args.out_dir)
    run_dir = out_base / f"{timestamp}_heuristic_feature_audit"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = extract_feature_rows(dataset_dir)
    if df.empty:
        raise RuntimeError("No feature rows extracted. Check dataset format/path.")

    feature_cols = [
        "p_evil_prior",
        "yes_on_failed_prior",
        "failed_team_count_prior",
        "success_team_count_prior",
        "talk_prior",
        "contradiction_rate_prior",
        "accuse_count_prior",
    ]
    target_col = "is_evil"

    udf = compute_univariate(df, feature_cols, target_col, n_perm=int(args.permutations), seed=int(args.seed))
    lpm = fit_linear_probability_model(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        seed=int(args.seed),
        n_boot=int(args.bootstrap),
    )
    vif_df = compute_vif(df, feature_cols)
    coeff_df = lpm["coefficients"]

    df.to_csv(run_dir / "feature_rows.csv", index=False)
    udf.to_csv(run_dir / "univariate_tests.csv", index=False)
    coeff_df.to_csv(run_dir / "linear_model_coefficients.csv", index=False)
    vif_df.to_csv(run_dir / "vif_diagnostics.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir.resolve()),
        "n_rows": int(len(df)),
        "n_games": int(df["game_id"].nunique()),
        "evil_ratio": float(df[target_col].mean()),
        "features": feature_cols,
        "analysis_notes": {
            "target": "is_evil (assassin or morgana)",
            "unit": "player snapshot before each quest within a game",
            "temporal_causality": "features use only information from earlier quests",
            "permutation_tests": int(args.permutations),
            "bootstrap_samples": int(args.bootstrap),
        },
        "linear_model": {
            "r2": lpm["r2"],
            "adjusted_r2": lpm["adjusted_r2"],
            "auc_from_linear_score": lpm["auc_from_linear_score"],
            "n_rows": lpm["n_rows"],
            "n_features": lpm["n_features"],
            "max_vif": float(vif_df["vif"].max()) if not vif_df.empty else 0.0,
        },
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plot_univariate_auc(udf, run_dir / "univariate_auc.html")
    save_plot_coefficients(coeff_df, run_dir / "linear_coefficients.html")
    viz_feats = [
        "p_evil_prior",
        "yes_on_failed_prior",
        "failed_team_count_prior",
        "talk_prior",
        "contradiction_rate_prior",
    ]
    save_plot_feature_box(df, viz_feats, run_dir / "feature_distributions.html")

    write_report_md(run_dir / "report.md", summary, udf, coeff_df)

    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit heuristic feature usefulness on Avalon dataset.")
    p.add_argument("--dataset-dir", type=str, default="avalon-nlu/dataset", help="Path to Avalon game JSON files")
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/analysis/heuristic_feature_audit",
        help="Output directory for analysis artifacts",
    )
    p.add_argument("--seed", type=int, default=17, help="Random seed")
    p.add_argument("--permutations", type=int, default=3000, help="Permutation count for univariate tests")
    p.add_argument("--bootstrap", type=int, default=1200, help="Bootstrap samples for coefficient CIs")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    run_dir = run_analysis(args)
    print(f"[ok] Heuristic feature audit saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
