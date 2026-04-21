#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repo root is importable when running as a script (python scripts/...)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.template_summary import build_memory_entries, load_game_json


def _canonicalize(role: str | None, one_servant: bool) -> str | None:
    if role is None:
        return None
    r = str(role).strip()
    if not r:
        return None
    if one_servant and r.lower() in {"servant-1", "servant-2", "servant"}:
        return "servant"
    return r


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _f1_by_label(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = _safe_div(float(tp), float(tp + fp))
        rec = _safe_div(float(tp), float(tp + fn))
        out[lab] = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) else 0.0
    return out


def _role_to_three_class(role: str | None, evil_roles: set[str], merlin_role: str) -> str | None:
    if role is None:
        return None
    if role == merlin_role:
        return "merlin"
    if role in evil_roles:
        return "evil"
    return "good"


def _three_class_metrics(y_true_roles: List[str], y_pred_roles: List[str]) -> Dict[str, Any]:
    labels = ["good", "evil", "merlin"]
    evil_roles = {"assassin", "morgana"}
    merlin_role = "merlin"

    y_true: List[str] = []
    y_pred: List[str] = []
    for t, p in zip(y_true_roles, y_pred_roles):
        t3 = _role_to_three_class(t, evil_roles, merlin_role)
        p3 = _role_to_three_class(p, evil_roles, merlin_role)
        if t3 is None:
            continue
        y_true.append(t3)
        y_pred.append(p3 if p3 is not None else "__other__")

    by_class = _f1_by_label(y_true, y_pred, labels)
    micro = _safe_div(float(sum(1 for t, p in zip(y_true, y_pred) if t == p)), float(len(y_true)))
    macro = float(np.mean([by_class[l] for l in labels])) if labels else 0.0
    return {
        "support": len(y_true),
        "micro_f1": micro,
        "macro_f1": macro,
        "f1_by_class": by_class,
    }


def _resolve_data_dir(data_dir: str | None, repo_root: Path) -> Path:
    if data_dir:
        p = Path(data_dir)
        if p.exists():
            return p
        raise FileNotFoundError(f"Data dir not found: {p}")

    c1 = repo_root / "dataset"
    c2 = repo_root / "avalon-nlu" / "dataset"
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    raise FileNotFoundError(f"Could not find dataset dir. Checked: {c1} and {c2}")


def _load_true_roles(files: List[Path], one_servant: bool) -> List[str]:
    roles: List[str] = []
    for f in files:
        game = load_game_json(f)
        entries = build_memory_entries(game, f.stem, memory_format="template")
        for e in entries:
            r = _canonicalize(e.proposer_role, one_servant=one_servant)
            if r is not None:
                roles.append(r)
    return roles


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute random-label baselines for proposer-role prediction.")
    ap.add_argument("--data_dir", type=str, default=None, help="Dataset directory with game JSON files")
    ap.add_argument("--runs", type=int, default=1000, help="Number of random runs")
    ap.add_argument("--seed", type=int, default=42, help="Base seed")
    ap.add_argument("--one-servant", action="store_true", help="Merge servant-1 and servant-2 to servant")
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _resolve_data_dir(args.data_dir, repo_root)
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {data_dir}")

    true_roles = _load_true_roles(files, one_servant=bool(args.one_servant))
    valid_roles = sorted(set(true_roles))
    if not valid_roles:
        raise RuntimeError("No proposer_role labels found in dataset.")

    per_role_runs: Dict[str, List[float]] = defaultdict(list)
    micro_runs: List[float] = []
    macro_runs: List[float] = []
    grouped_good_runs: List[float] = []
    grouped_evil_runs: List[float] = []
    grouped_merlin_runs: List[float] = []

    n = len(true_roles)
    for i in range(int(args.runs)):
        rng = np.random.default_rng(int(args.seed) + i)
        pred = rng.choice(valid_roles, size=n, replace=True).tolist()

        by_role = _f1_by_label(true_roles, pred, valid_roles)
        micro = _safe_div(float(sum(1 for t, p in zip(true_roles, pred) if t == p)), float(n))
        macro = float(np.mean([by_role[r] for r in valid_roles])) if valid_roles else 0.0

        grouped = _three_class_metrics(true_roles, pred)

        for r in valid_roles:
            per_role_runs[r].append(float(by_role[r]))
        micro_runs.append(float(micro))
        macro_runs.append(float(macro))
        grouped_good_runs.append(float(grouped["f1_by_class"]["good"]))
        grouped_evil_runs.append(float(grouped["f1_by_class"]["evil"]))
        grouped_merlin_runs.append(float(grouped["f1_by_class"]["merlin"]))

    result: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "runs": int(args.runs),
        "seed": int(args.seed),
        "one_servant": bool(args.one_servant),
        "support": n,
        "valid_roles": valid_roles,
        "role_distribution": {
            r: _safe_div(float(sum(1 for x in true_roles if x == r)), float(n)) for r in valid_roles
        },
        "overall": {
            "micro_f1_mean": float(np.mean(micro_runs)) if micro_runs else 0.0,
            "micro_f1_std": float(np.std(micro_runs)) if micro_runs else 0.0,
            "macro_f1_mean": float(np.mean(macro_runs)) if macro_runs else 0.0,
            "macro_f1_std": float(np.std(macro_runs)) if macro_runs else 0.0,
        },
        "f1_by_role_mean": {
            r: float(np.mean(per_role_runs[r])) if per_role_runs[r] else 0.0 for r in valid_roles
        },
        "grouped_3class": {
            "micro_f1_mean": float(np.mean(micro_runs)) if micro_runs else 0.0,
            "macro_f1_mean": float(np.mean([np.mean([grouped_good_runs[i], grouped_evil_runs[i], grouped_merlin_runs[i]]) for i in range(len(grouped_good_runs))])) if grouped_good_runs else 0.0,
            "f1_good_mean": float(np.mean(grouped_good_runs)) if grouped_good_runs else 0.0,
            "f1_evil_mean": float(np.mean(grouped_evil_runs)) if grouped_evil_runs else 0.0,
            "f1_merlin_mean": float(np.mean(grouped_merlin_runs)) if grouped_merlin_runs else 0.0,
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
