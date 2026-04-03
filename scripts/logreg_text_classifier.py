#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

# Ensure repo root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.svm_text_classifier import (
    _build_task,
    _build_text_features,
    _majority_label,
    _make_splitter,
    _stable_cues_from_folds,
    _top_lexical_cues,
    _write_summary_report,
    extract_text_samples,
)


def run_task_logreg(
    task_name: str,
    samples: List[Any],
    n_splits: int,
    seed: int,
    c_value: float,
    min_df: int,
    max_df: float,
    ngram_max: int,
    max_features: int,
    feature_mode: str,
    cues_top_k: int,
    cue_include_char_features: bool,
    cue_min_token_len: int,
    cue_min_fold_support: int,
) -> Dict[str, Any]:
    rows, y = _build_task(samples, task_name)
    if len(rows) == 0:
        return {"task": task_name, "error": "no_samples"}

    texts = np.array([r.text for r in rows], dtype=object)
    groups = np.array([r.game_id for r in rows], dtype=object)

    classes = np.unique(y)
    if classes.size < 2:
        return {"task": task_name, "error": "single_class_only", "n_samples": int(len(rows))}

    distinct_groups = len(set(str(g) for g in groups.tolist()))
    splits = int(max(2, min(n_splits, distinct_groups)))
    if splits < 2:
        return {"task": task_name, "error": "not_enough_groups", "n_samples": int(len(rows))}

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_pred_majority_all: List[int] = []
    fold_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    fold_cues: List[Dict[str, Any]] = []

    splitter = _make_splitter(y, groups, n_splits=splits, seed=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter, start=1):
        x_train = texts[train_idx].tolist()
        x_test = texts[test_idx].tolist()
        y_train = y[train_idx]
        y_test = y[test_idx]

        x_train_t, x_test_t, feat_names = _build_text_features(
            x_train=x_train,
            x_test=x_test,
            feature_mode=str(feature_mode),
            min_df=int(min_df),
            max_df=float(max_df),
            ngram_max=int(ngram_max),
            max_features=int(max_features),
        )

        clf = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=int(seed),
        )
        clf.fit(x_train_t, y_train)
        y_pred = clf.predict(x_test_t)

        fold_cues.append(
            {
                "fold": int(fold_idx),
                "cues": _top_lexical_cues(
                    feature_names=feat_names,
                    coefs=np.asarray(clf.coef_[0]),
                    top_k=int(cues_top_k),
                    include_char_features=bool(cue_include_char_features),
                    min_token_len=int(cue_min_token_len),
                ),
            }
        )

        majority = _majority_label(y_train)
        y_pred_majority = np.full_like(y_test, fill_value=majority)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_pred_majority_all.extend(y_pred_majority.tolist())

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "majority_accuracy": float(accuracy_score(y_test, y_pred_majority)),
                "majority_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_majority)),
                "majority_f1": float(f1_score(y_test, y_pred_majority, zero_division=0)),
            }
        )

        for local_i, idx in enumerate(test_idx):
            s = rows[int(idx)]
            pred_rows.append(
                {
                    "fold": fold_idx,
                    "game_id": s.game_id,
                    "quest": s.quest,
                    "player": s.player,
                    "true_role": s.role,
                    "true_label": int(y_test[local_i]),
                    "pred_label": int(y_pred[local_i]),
                }
            )

    y_true_arr = np.array(y_true_all, dtype=int)
    y_pred_arr = np.array(y_pred_all, dtype=int)
    y_pred_maj_arr = np.array(y_pred_majority_all, dtype=int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    cm_maj = confusion_matrix(y_true_arr, y_pred_maj_arr, labels=[0, 1])

    x_all_train, _, all_feat_names = _build_text_features(
        x_train=texts.tolist(),
        x_test=texts.tolist(),
        feature_mode=str(feature_mode),
        min_df=int(min_df),
        max_df=float(max_df),
        ngram_max=int(ngram_max),
        max_features=int(max_features),
    )
    clf_all = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=int(seed),
    )
    clf_all.fit(x_all_train, y)

    summary = {
        "task": task_name,
        "n_samples": int(len(rows)),
        "n_groups": int(distinct_groups),
        "n_splits": int(splits),
        "feature_mode": str(feature_mode),
        "model_type": "logistic_regression",
        "label_distribution": {
            "label_0": int((y_true_arr == 0).sum()),
            "label_1": int((y_true_arr == 1).sum()),
        },
        "model_metrics": {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        },
        "majority_baseline": {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_maj_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_maj_arr)),
            "f1": float(f1_score(y_true_arr, y_pred_maj_arr, zero_division=0)),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_majority": cm_maj.tolist(),
        "fold_metrics": fold_rows,
        "fold_lexical_cues": fold_cues,
        "lexical_cues": _top_lexical_cues(
            feature_names=all_feat_names,
            coefs=np.asarray(clf_all.coef_[0]),
            top_k=int(cues_top_k),
            include_char_features=bool(cue_include_char_features),
            min_token_len=int(cue_min_token_len),
        ),
        "stable_lexical_cues": _stable_cues_from_folds(
            fold_cues=fold_cues,
            min_fold_support=int(cue_min_fold_support),
            top_k=int(cues_top_k),
        ),
        "cue_filter": {
            "include_char_features": bool(cue_include_char_features),
            "min_token_len": int(cue_min_token_len),
            "min_fold_support": int(cue_min_fold_support),
        },
        "predictions": pred_rows,
    }
    return summary


def _write_task_outputs(task_result: Dict[str, Any], out_dir: Path) -> None:
    task = str(task_result.get("task", "task"))
    (out_dir / f"{task}_metrics.json").write_text(json.dumps(task_result, indent=2), encoding="utf-8")

    lexical = {
        "task": task,
        "feature_mode": task_result.get("feature_mode"),
        "model_type": task_result.get("model_type"),
        "label_distribution": task_result.get("label_distribution"),
        "global_lexical_cues": task_result.get("lexical_cues"),
        "stable_lexical_cues": task_result.get("stable_lexical_cues"),
        "fold_lexical_cues": task_result.get("fold_lexical_cues"),
        "cue_filter": task_result.get("cue_filter"),
    }
    (out_dir / f"{task}_lexical_cues.json").write_text(json.dumps(lexical, indent=2), encoding="utf-8")

    preds = task_result.get("predictions", [])
    if isinstance(preds, list) and preds:
        with (out_dir / f"{task}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["fold", "game_id", "quest", "player", "true_role", "true_label", "pred_label"],
            )
            writer.writeheader()
            for row in preds:
                writer.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train/evaluate logistic-regression text probes for Avalon role separability.")
    ap.add_argument("--dataset_dir", type=str, default="dataset", help="Directory containing Avalon game JSON files.")
    ap.add_argument("--out_dir", type=str, default="outputs/analysis/logreg_text_classifier", help="Base output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--folds", type=int, default=5, help="Maximum cross-validation folds.")
    ap.add_argument("--c", type=float, default=1.0, help="LogisticRegression C regularization.")
    ap.add_argument("--min_df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    ap.add_argument("--max_df", type=float, default=0.9, help="Maximum document frequency for TF-IDF.")
    ap.add_argument("--ngram_max", type=int, default=2, help="Maximum word n-gram size.")
    ap.add_argument("--max_features", type=int, default=5000, help="Maximum TF-IDF features.")
    ap.add_argument(
        "--feature_mode",
        type=str,
        default="word_char_tfidf",
        choices=["word_tfidf", "word_char_tfidf"],
        help="Text representation mode.",
    )
    ap.add_argument("--cues_top_k", type=int, default=20, help="Number of top lexical cues to save.")
    ap.add_argument(
        "--cue_include_char_features",
        action="store_true",
        help="Include char n-gram features in exported lexical cues (off by default for readability).",
    )
    ap.add_argument(
        "--cue_min_token_len",
        type=int,
        default=3,
        help="Minimum token length for word-based lexical cues.",
    )
    ap.add_argument(
        "--cue_min_fold_support",
        type=int,
        default=2,
        help="Minimum number of folds in which a cue must appear to be considered stable.",
    )
    return ap


def main(argv: List[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples = extract_text_samples(dataset_dir)
    if not samples:
        raise RuntimeError("No text samples extracted from dataset.")

    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = Path(args.out_dir) / f"{ts}_logreg_text_classifier"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_dir": str(dataset_dir),
        "seed": int(args.seed),
        "folds": int(args.folds),
        "c": float(args.c),
        "min_df": int(args.min_df),
        "max_df": float(args.max_df),
        "ngram_max": int(args.ngram_max),
        "max_features": int(args.max_features),
        "feature_mode": str(args.feature_mode),
        "cues_top_k": int(args.cues_top_k),
        "cue_include_char_features": bool(args.cue_include_char_features),
        "cue_min_token_len": int(args.cue_min_token_len),
        "cue_min_fold_support": int(args.cue_min_fold_support),
        "n_samples": int(len(samples)),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    tasks = [
        "good_vs_evil",
        "assassin_vs_morgana",
        "merlin_vs_other_good",
    ]
    results: List[Dict[str, Any]] = []
    for task_name in tasks:
        res = run_task_logreg(
            task_name=task_name,
            samples=samples,
            n_splits=int(args.folds),
            seed=int(args.seed),
            c_value=float(args.c),
            min_df=int(args.min_df),
            max_df=float(args.max_df),
            ngram_max=int(args.ngram_max),
            max_features=int(args.max_features),
            feature_mode=str(args.feature_mode),
            cues_top_k=int(args.cues_top_k),
            cue_include_char_features=bool(args.cue_include_char_features),
            cue_min_token_len=int(args.cue_min_token_len),
            cue_min_fold_support=int(args.cue_min_fold_support),
        )
        results.append(res)
        _write_task_outputs(res, run_dir)

    (run_dir / "all_tasks_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_summary_report(results, run_dir / "summary.md")

    print(f"Saved logistic text classifier outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
