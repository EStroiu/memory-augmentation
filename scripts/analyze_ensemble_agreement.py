#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class MethodRun:
    label: str
    source_path: Path
    config_path: Optional[Path]
    result_path: Path
    config: Dict[str, Any]
    cases: Dict[str, Dict[str, Any]]


@dataclass
class MethodBundle:
    label: str
    source_path: Path
    config_path: Optional[Path]
    config: Dict[str, Any]
    run_paths: List[Path]
    runs: List[MethodRun]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_result_path(source: Path, result_kind: str) -> Path:
    if source.is_file():
        return source

    current = source / "results_current.json"
    baseline = source / "results_baseline.json"

    if result_kind == "current":
        if current.exists():
            return current
        if baseline.exists():
            return baseline
    elif result_kind == "baseline":
        if baseline.exists():
            return baseline
        if current.exists():
            return current
    else:
        if current.exists():
            return current
        if baseline.exists():
            return baseline

    raise FileNotFoundError(f"No result JSON found under {source}")


def select_config_path(source: Path) -> Optional[Path]:
    if source.is_file():
        candidate = source.parent / "config.json"
        return candidate if candidate.exists() else None
    candidate = source / "config.json"
    return candidate if candidate.exists() else None


def select_run_result_path(run_dir: Path, result_kind: str) -> Path:
    current = run_dir / "results_current.json"
    baseline = run_dir / "results_baseline.json"

    if result_kind == "current":
        if current.exists():
            return current
        if baseline.exists():
            return baseline
    elif result_kind == "baseline":
        if baseline.exists():
            return baseline
        if current.exists():
            return current
    else:
        if current.exists():
            return current
        if baseline.exists():
            return baseline

    raise FileNotFoundError(f"No result JSON found in {run_dir}")


def infer_label(source: Path, result_path: Path, config: Dict[str, Any], explicit_label: Optional[str]) -> str:
    if explicit_label:
        return explicit_label

    if result_path.name == "results_current.json":
        label = config.get("current_prompt") or config.get("prompt_strategy")
        if isinstance(label, str) and label.strip() and label.strip() != "none":
            return label.strip()
    if result_path.name == "results_baseline.json":
        label = config.get("baseline_prompt") or config.get("prompt_strategy")
        if isinstance(label, str) and label.strip() and label.strip() != "none":
            return label.strip()

    if isinstance(config.get("current_prompt"), str) and config.get("current_prompt") not in {"", "none", None}:
        return str(config.get("current_prompt"))
    if isinstance(config.get("baseline_prompt"), str) and config.get("baseline_prompt") not in {"", "none", None}:
        return str(config.get("baseline_prompt"))
    return source.stem or result_path.stem


def load_run(source_text: str, label: Optional[str], result_kind: str) -> MethodRun:
    source = Path(source_text)
    result_path = select_result_path(source, result_kind)
    config_path = select_config_path(source)
    config: Dict[str, Any] = load_json(config_path) if config_path is not None else {}
    inferred_label = infer_label(source, result_path, config, label)
    cases = load_case_records(result_path, inferred_label)
    return MethodRun(
        label=inferred_label,
        source_path=source,
        config_path=config_path,
        result_path=result_path,
        config=config,
        cases=cases,
    )


def load_bundle(source_text: str, label: Optional[str], result_kind: str) -> MethodBundle:
    source = Path(source_text)
    config_path = select_config_path(source)
    config: Dict[str, Any] = load_json(config_path) if config_path is not None else {}
    inferred_label = infer_label(source, source, config, label)

    run_paths: List[Path] = []
    runs: List[MethodRun] = []

    if source.is_file():
        run = load_run(source_text, label, result_kind)
        return MethodBundle(
            label=run.label,
            source_path=source,
            config_path=config_path,
            config=config,
            run_paths=[run.result_path],
            runs=[run],
        )

    # If the folder directly contains a result file, treat it as a single-run bundle.
    direct_result = None
    try:
        direct_result = select_run_result_path(source, result_kind)
    except Exception:
        direct_result = None
    if direct_result is not None and direct_result.exists():
        run = load_run(str(direct_result), label, result_kind)
        return MethodBundle(
            label=run.label,
            source_path=source,
            config_path=config_path,
            config=config,
            run_paths=[run.result_path],
            runs=[run],
        )

    # Otherwise, treat the source as a top-level experiment folder with run_* subfolders.
    for child in sorted(source.glob("runs/run_*")):
        if child.is_dir():
            try:
                result_path = select_run_result_path(child, result_kind)
            except FileNotFoundError:
                continue
            run = load_run(str(result_path), label or inferred_label, result_kind)
            run_paths.append(result_path)
            runs.append(run)

    if not runs:
        raise FileNotFoundError(f"No runs found under {source}")

    return MethodBundle(
        label=inferred_label,
        source_path=source,
        config_path=config_path,
        config=config,
        run_paths=run_paths,
        runs=runs,
    )


def load_case_records(result_path: Path, method_label: str) -> Dict[str, Dict[str, Any]]:
    payload = load_json(result_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {result_path}")

    records: Dict[str, Dict[str, Any]] = {}
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue

        log_row = entry.get("log_row") or {}
        classification = entry.get("classification") or {}
        target = entry.get("target") or {}

        game_id = str(log_row.get("game_id") or target.get("game_id") or "").strip()
        entry_id = str(log_row.get("entry_id") or target.get("entry_id") or "").strip()
        quest_raw = log_row.get("quest", target.get("quest"))
        true_role = log_row.get("true_role") or classification.get("true_role")
        pred_role = log_row.get("pred_role") or classification.get("pred_role")

        if not game_id or not entry_id or quest_raw is None:
            continue
        if true_role is None or pred_role is None:
            continue

        key = f"{game_id}::{entry_id}"
        if key in records:
            raise ValueError(f"Duplicate case key {key} in {result_path}")

        records[key] = {
            "key": key,
            "game_id": game_id,
            "quest": int(quest_raw),
            "entry_id": entry_id,
            "true_role": str(true_role),
            "pred_role": str(pred_role),
            "method_label": method_label,
            "source_file": str(result_path),
            "index": idx,
        }

    return records


def _sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(row.get("game_id") or ""), int(row.get("quest") or 0), str(row.get("entry_id") or ""))


def _align_case_keys(runs: List[MethodRun], allow_partial_alignment: bool) -> Tuple[List[str], Dict[str, List[str]]]:
    key_sets = {run.label: set(run.cases.keys()) for run in runs}
    all_keys = set().union(*key_sets.values()) if key_sets else set()
    common_keys = set.intersection(*key_sets.values()) if key_sets else set()

    missing_by_method: Dict[str, List[str]] = {}
    for run in runs:
        missing = sorted(all_keys - key_sets[run.label])
        if missing:
            missing_by_method[run.label] = missing

    if not allow_partial_alignment and len(common_keys) != len(all_keys):
        details = "; ".join(f"{label}: missing {len(keys)}" for label, keys in missing_by_method.items())
        raise ValueError(
            "Input runs do not align on the same case keys. "
            f"Intersection={len(common_keys)} union={len(all_keys)}. {details}"
        )

    ordered = sorted(common_keys, key=lambda key: _sort_key(runs[0].cases[key]))
    return ordered, missing_by_method


def _label_universe(rows: Iterable[Dict[str, Any]], method_labels: List[str]) -> List[str]:
    labels = set()
    for row in rows:
        true_role = row.get("true_role")
        if true_role:
            labels.add(str(true_role))
        for label in method_labels:
            pred = row.get("predictions", {}).get(label)
            if pred:
                labels.add(str(pred))
        ensemble_pred = row.get("ensemble_pred")
        if ensemble_pred:
            labels.add(str(ensemble_pred))
    return sorted(labels)


def compute_confusion(true_labels: List[str], pred_labels: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    conf: Dict[str, Dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for truth, pred in zip(true_labels, pred_labels):
        conf.setdefault(truth, {p: 0 for p in labels})
        conf[truth][pred] = conf[truth].get(pred, 0) + 1
    return conf


def compute_metrics(true_labels: List[str], pred_labels: List[str], labels: List[str]) -> Dict[str, Any]:
    conf = compute_confusion(true_labels, pred_labels, labels)
    by_role: Dict[str, Dict[str, float]] = {}
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    correct = 0

    for label in labels:
        tp = float(conf.get(label, {}).get(label, 0))
        fp = float(sum(conf.get(other, {}).get(label, 0) for other in labels if other != label))
        fn = float(sum(conf.get(label, {}).get(other, 0) for other in labels if other != label))
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        by_role[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    for truth, pred in zip(true_labels, pred_labels):
        if truth == pred:
            correct += 1

    micro_prec = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_rec = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(v["f1"] for v in by_role.values()) / len(by_role) if by_role else 0.0

    return {
        "n_examples": len(true_labels),
        "accuracy": (correct / len(true_labels)) if true_labels else 0.0,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "by_role": by_role,
        "confusion": conf,
    }


def build_stacked_features(
    row: Dict[str, Any],
    method_labels: List[str],
    label_space: List[str],
) -> np.ndarray:
    predictions = row.get("predictions", {}) or {}
    counts = Counter(str(pred) for pred in predictions.values())
    features: List[float] = []

    # One-hot encode each method's prediction.
    for method in method_labels:
        pred = str(predictions.get(method, ""))
        for label in label_space:
            features.append(1.0 if pred == label else 0.0)

    # Count how many methods support each label.
    for label in label_space:
        features.append(float(counts.get(label, 0)))

    # Pairwise agreement indicators.
    for i, left in enumerate(method_labels):
        left_pred = str(predictions.get(left, ""))
        for right in method_labels[i + 1 :]:
            right_pred = str(predictions.get(right, ""))
            features.append(1.0 if left_pred == right_pred else 0.0)

    features.append(float(len(set(str(v) for v in predictions.values()))))
    return np.asarray(features, dtype=float)


def evaluate_stacked_ensemble(
    rows: List[Dict[str, Any]],
    method_labels: List[str],
    label_space: List[str],
) -> Dict[str, Any]:
    if not rows:
        return {
            "n_examples": 0,
            "accuracy": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "by_role": {},
            "confusion": {},
            "predictions": [],
        }

    groups = sorted({int(row.get("run_index", 0)) for row in rows if row.get("run_index") is not None})
    if len(groups) < 2:
        return {
            "n_examples": len(rows),
            "accuracy": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "by_role": {},
            "confusion": {},
            "predictions": [],
            "error": "not_enough_run_groups",
        }

    y_true = [str(row["true_role"]) for row in rows]
    classes = sorted(set(y_true) | set(label_space))
    feature_rows = np.vstack([build_stacked_features(row, method_labels, classes) for row in rows])
    run_indices = np.array([int(row.get("run_index", 0)) for row in rows], dtype=int)

    oof_pred: List[str] = [""] * len(rows)
    oof_prob_rows: List[List[float]] = [[0.0 for _ in classes] for _ in rows]

    for holdout_run in groups:
        train_mask = run_indices != holdout_run
        test_mask = run_indices == holdout_run
        if not np.any(train_mask) or not np.any(test_mask):
            continue

        x_train = feature_rows[train_mask]
        x_test = feature_rows[test_mask]
        y_train = np.array(np.array(y_true, dtype=object)[train_mask], dtype=object)

        if np.unique(y_train).size < 2:
            fallback = Counter(y_train.tolist()).most_common(1)[0][0]
            pred_labels = [str(fallback) for _ in range(int(test_mask.sum()))]
            for pos, label in zip(np.where(test_mask)[0].tolist(), pred_labels):
                oof_pred[pos] = label
                oof_prob_rows[pos] = [1.0 if c == fallback else 0.0 for c in classes]
            continue

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        clf.fit(x_train, y_train)
        pred_labels = clf.predict(x_test)
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(x_test)
        else:
            prob = np.zeros((x_test.shape[0], len(clf.classes_)), dtype=float)

        test_positions = np.where(test_mask)[0].tolist()
        for local_i, pos in enumerate(test_positions):
            oof_pred[pos] = str(pred_labels[local_i])
            prob_row = [0.0 for _ in classes]
            class_to_prob = {str(cls): float(prob[local_i][j]) for j, cls in enumerate(clf.classes_)}
            for j, label in enumerate(classes):
                prob_row[j] = class_to_prob.get(str(label), 0.0)
            oof_prob_rows[pos] = prob_row

    if any(pred == "" for pred in oof_pred):
        # Fill any gaps with deterministic majority vote over the three method predictions.
        for i, pred in enumerate(oof_pred):
            if pred:
                continue
            counts = Counter(str(rows[i].get("predictions", {}).get(m, "")) for m in method_labels)
            counts.pop("", None)
            oof_pred[i] = counts.most_common(1)[0][0] if counts else classes[0]
            oof_prob_rows[i] = [1.0 if c == oof_pred[i] else 0.0 for c in classes]

    metrics = compute_metrics(y_true, oof_pred, classes)
    return {
        **metrics,
        "predictions": oof_pred,
        "probabilities": oof_prob_rows,
        "classes": classes,
        "run_groups": groups,
        "feature_count": int(feature_rows.shape[1]),
        "mode": "stacked_logreg_leave_one_run_out",
    }


def choose_ensemble_label(
    predictions: Dict[str, str],
    method_order: List[str],
    per_method_role_f1: Dict[str, Dict[str, float]],
    per_method_macro_f1: Dict[str, float],
) -> Tuple[str, Dict[str, Any]]:
    counts = Counter(predictions.values())
    if not counts:
        return "", {"reason": "no_predictions"}

    top_count = max(counts.values())
    top_labels = sorted([label for label, count in counts.items() if count == top_count])
    vote_counts = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    if len(top_labels) == 1:
        chosen = top_labels[0]
        return chosen, {
            "reason": "majority_vote",
            "vote_counts": vote_counts,
            "top_count": top_count,
            "tied_labels": top_labels,
        }

    label_scores: Dict[str, float] = {}
    label_supporters: Dict[str, List[str]] = {}
    for label in top_labels:
        supporters = [method for method, pred in predictions.items() if pred == label]
        label_supporters[label] = supporters
        score = 0.0
        for method in supporters:
            score += float(per_method_role_f1.get(method, {}).get(label, 0.0))
        label_scores[label] = score

    best_score = max(label_scores.values())
    best_labels = [label for label, score in label_scores.items() if score == best_score]
    if len(best_labels) == 1:
        chosen = best_labels[0]
        return chosen, {
            "reason": "tie_break_role_f1",
            "vote_counts": vote_counts,
            "top_count": top_count,
            "tied_labels": top_labels,
            "tie_scores": label_scores,
            "supporters": label_supporters,
        }

    method_rank = {method: idx for idx, method in enumerate(method_order)}
    fallback_label = None
    fallback_tuple: Tuple[float, float, int, str] = (-1.0, -1.0, len(method_order), "")

    for label in best_labels:
        supporters = label_supporters.get(label, [])
        if supporters:
            best_supporter = max(supporters, key=lambda method: (per_method_macro_f1.get(method, 0.0), -method_rank.get(method, 0)))
            candidate = (
                float(per_method_macro_f1.get(best_supporter, 0.0)),
                float(per_method_role_f1.get(best_supporter, {}).get(label, 0.0)),
                -method_rank.get(best_supporter, len(method_order)),
                label,
            )
        else:
            candidate = (-1.0, -1.0, -method_rank.get(method_order[0], 0), label)

        if candidate > fallback_tuple:
            fallback_tuple = candidate
            fallback_label = label

    chosen = fallback_label if fallback_label is not None else best_labels[0]
    return chosen, {
        "reason": "deterministic_fallback",
        "vote_counts": vote_counts,
        "top_count": top_count,
        "tied_labels": top_labels,
        "tie_scores": label_scores,
        "supporters": label_supporters,
    }


def safe_accuracy(n_correct: int, n_total: int) -> float:
    return (n_correct / n_total) if n_total > 0 else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def validate_core_compatibility(runs: List[MethodRun]) -> None:
    if len(runs) < 2:
        return

    def core_signature(run: MethodRun) -> Dict[str, Any]:
        cfg = run.config or {}
        return {
            "data_dir": cfg.get("data_dir"),
            "max_games": cfg.get("max_games"),
            "grouped_role_task": cfg.get("grouped_role_task"),
            "one_servant": cfg.get("one_servant"),
            "memory_format": cfg.get("memory_format"),
        }

    first = core_signature(runs[0])
    mismatches = []
    for run in runs[1:]:
        sig = core_signature(run)
        if sig != first:
            mismatches.append((run.label, sig))
    if mismatches:
        lines = [f"{runs[0].label}: {first}"]
        for label, sig in mismatches:
            lines.append(f"{label}: {sig}")
        raise ValueError("Incompatible run configs for alignment:\n" + "\n".join(lines))


def build_role_accuracy_rows(
    rows: List[Dict[str, Any]],
    method_labels: List[str],
    metrics_by_method: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    roles = sorted({str(row["true_role"]) for row in rows})
    out: List[Dict[str, Any]] = []
    for role in roles:
        role_rows = [row for row in rows if row["true_role"] == role]
        total = len(role_rows)
        for method in method_labels:
            correct = sum(1 for row in role_rows if row["predictions"][method] == role)
            out.append(
                {
                    "true_role": role,
                    "method": method,
                    "n_cases": total,
                    "correct": correct,
                    "accuracy": safe_accuracy(correct, total),
                    "method_micro_f1": metrics_by_method[method]["micro_f1"],
                    "method_macro_f1": metrics_by_method[method]["macro_f1"],
                }
            )
    return out


def build_pairwise_agreement_rows(rows: List[Dict[str, Any]], method_labels: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = len(rows)
    for i, left in enumerate(method_labels):
        for right in method_labels[i + 1 :]:
            same = sum(1 for row in rows if row["predictions"][left] == row["predictions"][right])
            both_correct = sum(
                1
                for row in rows
                if row["predictions"][left] == row["true_role"] and row["predictions"][right] == row["true_role"]
            )
            agree_and_correct = sum(
                1
                for row in rows
                if row["predictions"][left] == row["predictions"][right] == row["true_role"]
            )
            out.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "n_cases": total,
                    "agreement_rate": safe_accuracy(same, total),
                    "both_correct_rate": safe_accuracy(both_correct, total),
                    "agree_and_correct_rate": safe_accuracy(agree_and_correct, total),
                }
            )
    return out


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out_lines = []
    out_lines.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out_lines.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        out_lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(out_lines)


def build_summary_markdown(
    runs: List[MethodRun],
    aligned_rows: List[Dict[str, Any]],
    method_metrics: Dict[str, Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    stacked_metrics: Dict[str, Any] | None,
    agreement_stats: Dict[str, Dict[str, Any]],
    role_accuracy_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
    missing_by_method: Dict[str, List[str]],
    weight_source: str,
) -> str:
    lines: List[str] = []
    lines.append("# Ensemble Agreement Summary")
    lines.append("")
    lines.append("## Inputs")
    for run in runs:
        lines.append(f"- {run.label}: {run.result_path}")
    lines.append(f"- Weight source: {weight_source}")
    if missing_by_method:
        lines.append("- Alignment warnings:")
        for label, missing in missing_by_method.items():
            lines.append(f"  - {label}: missing {len(missing)} cases from the union")
    lines.append("")

    lines.append("## Overall Metrics")
    metric_rows: List[List[str]] = []
    for run in runs:
        metrics = method_metrics[run.label]
        metric_rows.append(
            [
                run.label,
                str(metrics["n_examples"]),
                fmt_pct(metrics["accuracy"]),
                fmt_pct(metrics["micro_f1"]),
                fmt_pct(metrics["macro_f1"]),
            ]
        )
    metric_rows.append(
        [
            "ensemble",
            str(ensemble_metrics["n_examples"]),
            fmt_pct(ensemble_metrics["accuracy"]),
            fmt_pct(ensemble_metrics["micro_f1"]),
            fmt_pct(ensemble_metrics["macro_f1"]),
        ]
    )
    if stacked_metrics is not None:
        metric_rows.append(
            [
                "stacked_logreg",
                str(stacked_metrics.get("n_examples", 0)),
                fmt_pct(float(stacked_metrics.get("accuracy", 0.0))),
                fmt_pct(float(stacked_metrics.get("micro_f1", 0.0))),
                fmt_pct(float(stacked_metrics.get("macro_f1", 0.0))),
            ]
        )
    lines.append(format_table(["method", "n", "accuracy", "micro_f1", "macro_f1"], metric_rows))
    lines.append("")

    lines.append("## Agreement Buckets")
    bucket_rows: List[List[str]] = []
    for bucket_name, stats in agreement_stats.items():
        bucket_rows.append(
            [
                bucket_name,
                str(stats["count"]),
                fmt_pct(stats["ensemble_accuracy"]),
                fmt_pct(stats["oracle_accuracy"]),
                fmt_pct(stats["any_method_accuracy"]),
            ]
        )
    lines.append(format_table(["bucket", "count", "ensemble_acc", "oracle_acc", "any_method_acc"], bucket_rows))
    lines.append("")

    lines.append("## Role Specialists")
    role_rows: List[List[str]] = []
    grouped_by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in role_accuracy_rows:
        grouped_by_role[str(row["true_role"])].append(row)
    for role in sorted(grouped_by_role.keys()):
        ranked = sorted(grouped_by_role[role], key=lambda row: (-float(row["accuracy"]), str(row["method"])))
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        role_rows.append(
            [
                role,
                str(best["method"]),
                fmt_pct(float(best["accuracy"])),
                f"{str(second['method'])} ({fmt_pct(float(second['accuracy']))})" if second else "-",
            ]
        )
    lines.append(format_table(["role", "best_method", "best_acc", "runner_up"], role_rows))
    lines.append("")

    lines.append("## Pairwise Agreement")
    pair_rows: List[List[str]] = []
    for row in pairwise_rows:
        pair_rows.append(
            [
                f"{row['left_method']} vs {row['right_method']}",
                str(row["n_cases"]),
                fmt_pct(float(row["agreement_rate"])),
                fmt_pct(float(row["both_correct_rate"])),
                fmt_pct(float(row["agree_and_correct_rate"])),
            ]
        )
    lines.append(format_table(["pair", "n", "agree_rate", "both_correct", "agree_and_correct"], pair_rows))
    lines.append("")

    lines.append("## Notes")
    lines.append("- The tie-break weights are derived from the same aligned cases in this exploratory pass.")
    if stacked_metrics is not None:
        lines.append("- Stacked_logreg is evaluated with leave-one-run-out cross-validation over the existing runs.")
    lines.append("- Use a held-out calibration split before treating the ensemble as a final decision rule.")
    lines.append("- Full-role runs are preferable here; grouped-role runs are useful only as a smoke test for the ensemble code.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Align multiple role-prediction runs and build an agreement-aware ensemble.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Run directories or result JSON files to compare.",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional method labels in the same order as --inputs.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where ensemble outputs will be written.",
    )
    ap.add_argument(
        "--result_kind",
        type=str,
        default="auto",
        choices=["auto", "current", "baseline"],
        help="Which result file to load when an input is a run directory.",
    )
    ap.add_argument(
        "--allow_partial_alignment",
        action="store_true",
        help="Use the intersection of case keys instead of requiring full alignment.",
    )
    args = ap.parse_args(argv)

    if args.labels and len(args.labels) != len(args.inputs):
        print("--labels must match --inputs in length", file=sys.stderr)
        return 1

    bundles: List[MethodBundle] = []
    for idx, input_text in enumerate(args.inputs):
        label = args.labels[idx] if args.labels else None
        bundles.append(load_bundle(input_text, label, args.result_kind))

    if not bundles:
        print("No inputs provided", file=sys.stderr)
        return 1

    run_counts = {bundle.label: len(bundle.runs) for bundle in bundles}
    unique_run_counts = sorted(set(run_counts.values()))
    if len(unique_run_counts) != 1:
        details = ", ".join(f"{label}={count}" for label, count in sorted(run_counts.items()))
        raise ValueError(f"All inputs must have the same number of runs. Got: {details}")
    run_count = unique_run_counts[0]

    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    method_labels = [bundle.label for bundle in bundles]
    for bundle in bundles:
        validate_core_compatibility(bundle.runs)

    aligned_rows: List[Dict[str, Any]] = []
    source_run_count = run_count
    alignment_missing_by_method: Dict[str, List[str]] = defaultdict(list)
    for run_index in range(source_run_count):
        runs = [bundle.runs[run_index] for bundle in bundles]
        aligned_keys, missing_by_method = _align_case_keys(runs, allow_partial_alignment=bool(args.allow_partial_alignment))
        for label, missing_keys in missing_by_method.items():
            alignment_missing_by_method[label].extend([f"run{run_index + 1}:{key}" for key in missing_keys])

        for key in aligned_keys:
            first = runs[0].cases[key]
            true_role = str(first["true_role"])
            predictions = {run.label: str(run.cases[key]["pred_role"]) for run in runs}

            # Sanity check: all methods should agree on the truth label for a case.
            mismatched_truths = sorted({str(run.cases[key]["true_role"]) for run in runs})
            if len(mismatched_truths) != 1:
                raise ValueError(f"True-role mismatch for case {key}: {mismatched_truths}")

            aligned_rows.append(
                {
                    "key": f"run{run_index + 1}::{key}",
                    "run_index": run_index + 1,
                    "game_id": first["game_id"],
                    "quest": first["quest"],
                    "entry_id": first["entry_id"],
                    "true_role": true_role,
                    "predictions": predictions,
                }
            )

    # Individual metrics for each method on the aligned cases.
    method_metrics: Dict[str, Dict[str, Any]] = {}
    true_labels = [str(row["true_role"]) for row in aligned_rows]
    for method in method_labels:
        preds = [str(row["predictions"][method]) for row in aligned_rows]
        method_metrics[method] = compute_metrics(true_labels, preds, sorted(set(true_labels + preds)))

    per_method_role_f1 = {
        method: {role: float(stats["f1"]) for role, stats in method_metrics[method]["by_role"].items()}
        for method in method_labels
    }
    per_method_macro_f1 = {method: float(method_metrics[method]["macro_f1"]) for method in method_labels}

    ensemble_rows: List[Dict[str, Any]] = []
    agreement_counters: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "ensemble_correct": 0, "oracle_correct": 0, "any_method_correct": 0})

    ensemble_true: List[str] = []
    ensemble_pred: List[str] = []
    for row in aligned_rows:
        true_role = str(row["true_role"])
        predictions = {method: str(pred) for method, pred in row["predictions"].items()}
        counts = Counter(predictions.values())
        unique_count = len(counts)
        pattern = "-".join(str(counts[c]) for c in sorted(counts.values(), reverse=True)) if counts else "0"
        majority_label, decision_meta = choose_ensemble_label(predictions, method_labels, per_method_role_f1, per_method_macro_f1)
        vote_counts = decision_meta.get("vote_counts", {})

        correct_methods = [method for method, pred in predictions.items() if pred == true_role]
        any_correct = bool(correct_methods)
        ensemble_correct = majority_label == true_role
        oracle_correct = any_correct

        if unique_count == 1:
            agreement_bucket = "unanimous"
        elif unique_count == len(method_labels):
            agreement_bucket = "full_disagreement"
        elif max(counts.values()) > (len(method_labels) / 2.0):
            agreement_bucket = "majority"
        else:
            agreement_bucket = f"{unique_count}_unique"

        agreement_counters[agreement_bucket]["count"] += 1
        agreement_counters[agreement_bucket]["ensemble_correct"] += 1 if ensemble_correct else 0
        agreement_counters[agreement_bucket]["oracle_correct"] += 1 if oracle_correct else 0
        agreement_counters[agreement_bucket]["any_method_correct"] += 1 if any_correct else 0

        ensemble_true.append(true_role)
        ensemble_pred.append(majority_label)

        ensemble_rows.append(
            {
                "key": row["key"],
                "run_index": row.get("run_index"),
                "game_id": row["game_id"],
                "quest": row["quest"],
                "entry_id": row["entry_id"],
                "true_role": true_role,
                "predictions": predictions,
                "vote_counts": dict(vote_counts),
                "vote_pattern": pattern,
                "unique_prediction_count": unique_count,
                "agreement_bucket": agreement_bucket,
                "ensemble_pred": majority_label,
                "ensemble_correct": ensemble_correct,
                "oracle_correct": oracle_correct,
                "any_method_correct": any_correct,
                "correct_methods": correct_methods,
                "decision_reason": decision_meta.get("reason"),
                "decision_meta": decision_meta,
            }
        )

    labels = _label_universe(ensemble_rows, method_labels)
    ensemble_metrics = compute_metrics(ensemble_true, ensemble_pred, labels)
    stacked_metrics = evaluate_stacked_ensemble(ensemble_rows, method_labels, labels)
    for row, pred, prob in zip(
        ensemble_rows,
        stacked_metrics.get("predictions", []),
        stacked_metrics.get("probabilities", []),
    ):
        row["stacked_pred"] = str(pred)
        row["stacked_correct"] = str(pred) == str(row["true_role"])
        row["stacked_probabilities"] = [float(x) for x in prob]

    # Agreement summary.
    agreement_stats: Dict[str, Dict[str, Any]] = {}
    for bucket, stats in agreement_counters.items():
        count = int(stats["count"])
        agreement_stats[bucket] = {
            "count": count,
            "ensemble_correct": int(stats["ensemble_correct"]),
            "oracle_correct": int(stats["oracle_correct"]),
            "any_method_correct": int(stats["any_method_correct"]),
            "ensemble_accuracy": safe_accuracy(int(stats["ensemble_correct"]), count),
            "oracle_accuracy": safe_accuracy(int(stats["oracle_correct"]), count),
            "any_method_accuracy": safe_accuracy(int(stats["any_method_correct"]), count),
        }

    role_accuracy_rows = build_role_accuracy_rows(ensemble_rows, method_labels, method_metrics)
    pairwise_rows = build_pairwise_agreement_rows(ensemble_rows, method_labels)

    # Add best method per role summary.
    role_best: Dict[str, Dict[str, Any]] = {}
    for role in sorted({str(row["true_role"]) for row in ensemble_rows}):
        role_rows = [row for row in role_accuracy_rows if row["true_role"] == role]
        role_rows = sorted(role_rows, key=lambda row: (-float(row["accuracy"]), str(row["method"])))
        role_best[role] = {
            "best_method": role_rows[0]["method"],
            "best_accuracy": float(role_rows[0]["accuracy"]),
            "runner_up_method": role_rows[1]["method"] if len(role_rows) > 1 else None,
            "runner_up_accuracy": float(role_rows[1]["accuracy"]) if len(role_rows) > 1 else None,
        }

    method_global_accuracy = {
        method: float(method_metrics[method]["accuracy"]) for method in method_labels
    }

    # Build detailed rows for CSV output.
    csv_rows: List[Dict[str, Any]] = []
    for row in ensemble_rows:
        base = {
            "key": row["key"],
            "game_id": row["game_id"],
            "quest": row["quest"],
            "entry_id": row["entry_id"],
            "true_role": row["true_role"],
            "vote_pattern": row["vote_pattern"],
            "unique_prediction_count": row["unique_prediction_count"],
            "agreement_bucket": row["agreement_bucket"],
            "ensemble_pred": row["ensemble_pred"],
            "ensemble_correct": row["ensemble_correct"],
            "stacked_pred": row.get("stacked_pred", ""),
            "stacked_correct": row.get("stacked_correct", ""),
            "oracle_correct": row["oracle_correct"],
            "any_method_correct": row["any_method_correct"],
            "correct_methods": "|".join(row["correct_methods"]),
            "decision_reason": row["decision_reason"],
            "decision_meta_json": json.dumps(row["decision_meta"], ensure_ascii=False),
            "stacked_probabilities_json": json.dumps(row.get("stacked_probabilities", []), ensure_ascii=False),
            "predictions_json": json.dumps(row["predictions"], ensure_ascii=False),
            "vote_counts_json": json.dumps(row["vote_counts"], ensure_ascii=False),
        }
        for method in method_labels:
            base[f"{method}_pred"] = row["predictions"][method]
            base[f"{method}_correct"] = row["predictions"][method] == row["true_role"]
        csv_rows.append(base)

    # Persist artifacts.
    dump_json(outdir / "ensemble_predictions.json", ensemble_rows)
    dump_json(
        outdir / "ensemble_aggregate.json",
        {
            "inputs": [
                {
                    "label": bundle.label,
                    "source_path": str(bundle.source_path),
                    "config_path": str(bundle.config_path) if bundle.config_path else None,
                    "config_core": {
                        "data_dir": bundle.config.get("data_dir"),
                        "max_games": bundle.config.get("max_games"),
                        "grouped_role_task": bundle.config.get("grouped_role_task"),
                        "one_servant": bundle.config.get("one_servant"),
                        "memory_format": bundle.config.get("memory_format"),
                    },
                    "run_count": len(bundle.runs),
                    "run_paths": [str(path) for path in bundle.run_paths],
                }
                for bundle in bundles
            ],
            "weight_source": "same_aligned_cases",
            "n_cases": len(ensemble_rows),
            "method_labels": method_labels,
            "method_metrics": method_metrics,
            "method_global_accuracy": method_global_accuracy,
            "ensemble_metrics": ensemble_metrics,
            "stacked_metrics": stacked_metrics,
            "agreement_stats": agreement_stats,
            "role_best_method": role_best,
            "pairwise_agreement": pairwise_rows,
            "alignment_missing_by_method": dict(alignment_missing_by_method),
        },
    )

    write_csv(
        outdir / "agreement_table.csv",
        csv_rows,
        [
            "key",
            "game_id",
            "quest",
            "entry_id",
            "true_role",
            "vote_pattern",
            "unique_prediction_count",
            "agreement_bucket",
            "ensemble_pred",
            "ensemble_correct",
            "stacked_pred",
            "stacked_correct",
            "oracle_correct",
            "any_method_correct",
            "correct_methods",
            "decision_reason",
            "decision_meta_json",
            "stacked_probabilities_json",
            "predictions_json",
            "vote_counts_json",
            *[f"{method}_pred" for method in method_labels],
            *[f"{method}_correct" for method in method_labels],
        ],
    )

    disagreement_rows = [row for row in csv_rows if row["agreement_bucket"] != "unanimous" or not row["ensemble_correct"]]
    write_csv(
        outdir / "disagreement_cases.csv",
        disagreement_rows,
        [
            "key",
            "game_id",
            "quest",
            "entry_id",
            "true_role",
            "agreement_bucket",
            "vote_pattern",
            "ensemble_pred",
            "ensemble_correct",
            "stacked_pred",
            "stacked_correct",
            "oracle_correct",
            "correct_methods",
            "decision_reason",
            "decision_meta_json",
            "stacked_probabilities_json",
            "predictions_json",
            *[f"{method}_pred" for method in method_labels],
        ],
    )

    # Serialize method metrics as CSV-friendly rows in a separate, flat file.
    method_metric_csv_rows: List[Dict[str, Any]] = []
    for method in method_labels:
        metrics = method_metrics[method]
        method_metric_csv_rows.append(
            {
                "method": method,
                "role": "__overall__",
                "precision": "",
                "recall": "",
                "f1": "",
                "accuracy": metrics["accuracy"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "n_examples": metrics["n_examples"],
            }
        )
        for role, stats in sorted(metrics["by_role"].items()):
            method_metric_csv_rows.append(
                {
                    "method": method,
                    "role": role,
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1"],
                    "accuracy": "",
                    "micro_f1": "",
                    "macro_f1": "",
                    "n_examples": "",
                }
            )

    write_csv(
        outdir / "method_metrics.csv",
        method_metric_csv_rows,
        ["method", "role", "precision", "recall", "f1", "accuracy", "micro_f1", "macro_f1", "n_examples"],
    )
    write_csv(
        outdir / "role_method_accuracy.csv",
        role_accuracy_rows,
        [
            "true_role",
            "method",
            "n_cases",
            "correct",
            "accuracy",
            "method_micro_f1",
            "method_macro_f1",
        ],
    )
    write_csv(
        outdir / "pairwise_agreement.csv",
        pairwise_rows,
        [
            "left_method",
            "right_method",
            "n_cases",
            "agreement_rate",
            "both_correct_rate",
            "agree_and_correct_rate",
        ],
    )

    summary_md = build_summary_markdown(
        runs=runs,
        aligned_rows=ensemble_rows,
        method_metrics=method_metrics,
        ensemble_metrics=ensemble_metrics,
        stacked_metrics=stacked_metrics,
        agreement_stats=agreement_stats,
        role_accuracy_rows=role_accuracy_rows,
        pairwise_rows=pairwise_rows,
        missing_by_method=missing_by_method,
        weight_source="same_aligned_cases",
    )
    (outdir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Saved ensemble analysis to: {outdir}")
    print(
        f"Ensemble micro-F1={ensemble_metrics['micro_f1']:.3f} | "
        f"accuracy={ensemble_metrics['accuracy']:.3f} | n={ensemble_metrics['n_examples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())