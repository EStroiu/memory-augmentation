#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None  # type: ignore


EVIL_ROLES = {"assassin", "morgana"}
GOOD_ROLES = {"percival", "servant-1", "servant-2"}


@dataclass
class TextSample:
    game_id: str
    quest: int
    player: str
    role: str
    text: str


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def extract_text_samples(dataset_dir: Path) -> List[TextSample]:
    samples: List[TextSample] = []

    for path in sorted(dataset_dir.glob("*.json"), key=lambda p: p.name):
        game = load_json(path)
        game_id = path.stem

        users = game.get("users", {}) or {}
        role_by_name: Dict[str, str] = {}
        for u in users.values():
            name = u.get("name")
            role = u.get("role")
            if isinstance(name, str) and isinstance(role, str):
                role_by_name[name] = role

        grouped: Dict[Tuple[int, str], List[str]] = {}
        for msg in (game.get("messages", {}) or {}).values():
            player = msg.get("player")
            if not isinstance(player, str) or player == "system":
                continue
            role = role_by_name.get(player)
            if not isinstance(role, str):
                continue
            quest = _safe_int(msg.get("quest"), default=0)
            text = str(msg.get("msg") or "").strip()
            if not text:
                continue
            grouped.setdefault((quest, player), []).append(text)

        for (quest, player), utterances in grouped.items():
            role = role_by_name.get(player)
            if not isinstance(role, str):
                continue
            merged = " ".join(utterances).strip()
            if not merged:
                continue
            samples.append(
                TextSample(
                    game_id=game_id,
                    quest=int(quest),
                    player=player,
                    role=role,
                    text=merged,
                )
            )

    return samples


def _build_task(samples: Iterable[TextSample], task_name: str) -> Tuple[List[TextSample], np.ndarray]:
    rows = list(samples)

    if task_name == "good_vs_evil":
        filtered = [r for r in rows if r.role in GOOD_ROLES or r.role in EVIL_ROLES]
        y = np.array([1 if r.role in EVIL_ROLES else 0 for r in filtered], dtype=int)
        return filtered, y

    if task_name == "assassin_vs_morgana":
        filtered = [r for r in rows if r.role in {"assassin", "morgana"}]
        y = np.array([1 if r.role == "morgana" else 0 for r in filtered], dtype=int)
        return filtered, y

    if task_name == "merlin_vs_other_good":
        filtered = [r for r in rows if r.role == "merlin" or r.role in GOOD_ROLES]
        y = np.array([1 if r.role == "merlin" else 0 for r in filtered], dtype=int)
        return filtered, y

    raise ValueError(f"Unknown task: {task_name}")


def _make_splitter(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y), dtype=int), y, groups
        )
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(y), dtype=int), y, groups)


def _majority_label(y_train: np.ndarray) -> int:
    vals, counts = np.unique(y_train, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _is_informative_word_feature(feature: str, min_token_len: int) -> bool:
    phrase = feature[2:].strip().lower()
    if not phrase:
        return False

    tokens = [tok for tok in phrase.split() if tok]
    if not tokens:
        return False

    content_tokens = [
        tok for tok in tokens if tok.isalpha() and len(tok) >= int(min_token_len) and tok not in ENGLISH_STOP_WORDS
    ]
    return bool(content_tokens)


def _feature_is_allowed(feature: str, include_char_features: bool, min_token_len: int) -> bool:
    if feature.startswith("c:"):
        return bool(include_char_features)
    if feature.startswith("w:"):
        return _is_informative_word_feature(feature, min_token_len=int(min_token_len))
    return False


def _top_lexical_cues(
    feature_names: np.ndarray,
    coefs: np.ndarray,
    top_k: int = 20,
    include_char_features: bool = False,
    min_token_len: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    if feature_names.size == 0 or coefs.size == 0:
        return {"positive": [], "negative": []}

    coef = np.asarray(coefs, dtype=float).reshape(-1)
    allowed_idx = np.array(
        [
            i
            for i, name in enumerate(feature_names.tolist())
            if _feature_is_allowed(str(name), bool(include_char_features), int(min_token_len))
        ],
        dtype=int,
    )

    if allowed_idx.size == 0:
        allowed_idx = np.arange(coef.size, dtype=int)

    allowed_coefs = coef[allowed_idx]
    k = int(max(1, min(int(top_k), allowed_idx.size)))

    pos_local = np.argsort(allowed_coefs)[-k:][::-1]
    neg_local = np.argsort(allowed_coefs)[:k]
    pos_idx = allowed_idx[pos_local]
    neg_idx = allowed_idx[neg_local]
    return {
        "positive": [
            {"feature": str(feature_names[i]), "weight": float(coef[i])}
            for i in pos_idx
        ],
        "negative": [
            {"feature": str(feature_names[i]), "weight": float(coef[i])}
            for i in neg_idx
        ],
    }


def _stable_cues_from_folds(
    fold_cues: List[Dict[str, Any]],
    min_fold_support: int,
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    support = max(1, int(min_fold_support))

    feature_counts: Dict[str, Counter] = {
        "positive": Counter(),
        "negative": Counter(),
    }
    feature_weights: Dict[str, Dict[str, List[float]]] = {
        "positive": {},
        "negative": {},
    }

    for fold_item in fold_cues:
        cues = fold_item.get("cues", {}) or {}
        for side in ("positive", "negative"):
            items = cues.get(side, []) or []
            seen_this_fold = set()
            for item in items:
                feat = str(item.get("feature", "")).strip()
                if not feat:
                    continue
                if feat not in seen_this_fold:
                    feature_counts[side][feat] += 1
                    seen_this_fold.add(feat)
                feature_weights[side].setdefault(feat, []).append(float(item.get("weight", 0.0)))

    stable: Dict[str, List[Dict[str, Any]]] = {"positive": [], "negative": []}
    for side in ("positive", "negative"):
        rows: List[Dict[str, Any]] = []
        for feat, count in feature_counts[side].items():
            if int(count) < support:
                continue
            weights = feature_weights[side].get(feat, [0.0])
            rows.append(
                {
                    "feature": feat,
                    "mean_weight": float(np.mean(np.asarray(weights, dtype=float))),
                    "fold_support": int(count),
                }
            )
        rows.sort(key=lambda r: abs(float(r["mean_weight"])), reverse=True)
        stable[side] = rows[: int(max(1, top_k))]
    return stable


def _build_text_features(
    x_train: List[str],
    x_test: List[str],
    feature_mode: str,
    min_df: int,
    max_df: float,
    ngram_max: int,
    max_features: int,
) -> Tuple[Any, Any, np.ndarray]:
    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, int(max(1, ngram_max))),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{1,}\b",
        min_df=int(max(1, min_df)),
        max_df=float(max_df),
        max_features=int(max(200, max_features)),
        sublinear_tf=True,
    )
    x_train_word = word_vec.fit_transform(x_train)
    x_test_word = word_vec.transform(x_test)
    word_names = np.array([f"w:{n}" for n in word_vec.get_feature_names_out()], dtype=object)

    if str(feature_mode) == "word_tfidf":
        return x_train_word, x_test_word, word_names

    # Char-level ngrams often capture stylistic/deception patterns missed by pure words.
    char_features = max(500, int(max_features // 2))
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=int(max(1, min_df)),
        max_df=float(max_df),
        max_features=int(char_features),
        sublinear_tf=True,
    )
    x_train_char = char_vec.fit_transform(x_train)
    x_test_char = char_vec.transform(x_test)
    char_names = np.array([f"c:{n}" for n in char_vec.get_feature_names_out()], dtype=object)

    x_train_all = hstack([x_train_word, x_train_char], format="csr")
    x_test_all = hstack([x_test_word, x_test_char], format="csr")
    all_names = np.concatenate([word_names, char_names])
    return x_train_all, x_test_all, all_names


def run_task(
    task_name: str,
    samples: List[TextSample],
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

        clf = LinearSVC(C=float(c_value), class_weight="balanced")
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
    clf_all = LinearSVC(C=float(c_value), class_weight="balanced")
    clf_all.fit(x_all_train, y)
    global_cues = _top_lexical_cues(
        feature_names=all_feat_names,
        coefs=np.asarray(clf_all.coef_[0]),
        top_k=int(cues_top_k),
        include_char_features=bool(cue_include_char_features),
        min_token_len=int(cue_min_token_len),
    )
    stable_cues = _stable_cues_from_folds(
        fold_cues=fold_cues,
        min_fold_support=int(cue_min_fold_support),
        top_k=int(cues_top_k),
    )

    summary = {
        "task": task_name,
        "n_samples": int(len(rows)),
        "n_groups": int(distinct_groups),
        "n_splits": int(splits),
        "feature_mode": str(feature_mode),
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
        "lexical_cues": global_cues,
        "stable_lexical_cues": stable_cues,
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



def _write_summary_report(results: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# SVM Text Classifier Report")
    lines.append("")
    lines.append("This report evaluates text-only role separability using TF-IDF + Linear SVM with grouped CV by game.")
    lines.append("")

    for res in results:
        task = str(res.get("task", "unknown"))
        lines.append(f"## {task}")
        if "error" in res:
            lines.append(f"- status: {res['error']}")
            lines.append("")
            continue
        mm = res.get("model_metrics", {}) or {}
        mb = res.get("majority_baseline", {}) or {}
        lines.append(f"- samples: {res.get('n_samples', 0)}")
        lines.append(f"- groups: {res.get('n_groups', 0)} | folds: {res.get('n_splits', 0)}")
        lines.append(
            f"- model: accuracy={float(mm.get('accuracy', 0.0)):.4f}, "
            f"balanced_accuracy={float(mm.get('balanced_accuracy', 0.0)):.4f}, f1={float(mm.get('f1', 0.0)):.4f}"
        )
        lines.append(
            f"- majority: accuracy={float(mb.get('accuracy', 0.0)):.4f}, "
            f"balanced_accuracy={float(mb.get('balanced_accuracy', 0.0)):.4f}, f1={float(mb.get('f1', 0.0)):.4f}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train/evaluate text-only SVM probes for Avalon role separability.")
    ap.add_argument("--dataset_dir", type=str, default="dataset", help="Directory containing Avalon game JSON files.")
    ap.add_argument("--out_dir", type=str, default="outputs/analysis/svm_text_classifier", help="Base output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--folds", type=int, default=5, help="Maximum cross-validation folds.")
    ap.add_argument("--c", type=float, default=1.0, help="LinearSVC C regularization.")
    ap.add_argument("--min_df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    ap.add_argument("--max_df", type=float, default=0.9, help="Maximum document frequency for TF-IDF.")
    ap.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size (1 means unigrams only).")
    ap.add_argument("--max_features", type=int, default=5000, help="Maximum TF-IDF features.")
    ap.add_argument(
        "--feature_mode",
        type=str,
        default="word_char_tfidf",
        choices=["word_tfidf", "word_char_tfidf"],
        help="Text representation mode. 'word_char_tfidf' is usually stronger than pure words.",
    )
    ap.add_argument("--cues_top_k", type=int, default=20, help="Number of top lexical cues to save per class direction.")
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
    run_dir = Path(args.out_dir) / f"{ts}_svm_text_classifier"
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
        res = run_task(
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

    print(f"Saved SVM text classifier outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
