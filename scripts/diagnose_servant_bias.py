#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


VALID_ROLES = ["assassin", "merlin", "morgana", "percival", "servant-1", "servant-2"]


def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * num / den


def _map_merged_servant(role: str | None) -> str | None:
    if role in {"servant-1", "servant-2"}:
        return "servant"
    return role


def _parse_json_role(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and isinstance(obj.get("role"), str):
            return obj["role"].strip()
    except Exception:
        pass
    try:
        match = re.search(r"\{[^{}]*\}", stripped, flags=re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("role"), str):
                return obj["role"].strip()
    except Exception:
        pass
    return None


def _extract_role_like_pipeline(text: str | None) -> str | None:
    role = _parse_json_role(text)
    if role:
        for valid in VALID_ROLES:
            if role.lower() == valid.lower():
                return valid
    if not text:
        return None
    lower = text.lower()
    hits = [role_name for role_name in VALID_ROLES if role_name in lower]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return sorted(hits, key=len, reverse=True)[0]
    for role_name in VALID_ROLES:
        if re.search(rf"\b{re.escape(role_name)}\b", text, flags=re.IGNORECASE):
            return role_name
    return None


def _iter_result_rows(eval_dir: Path, variant: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    runs_dir = eval_dir / "runs"
    if not runs_dir.exists():
        return
    file_name = f"results_{variant}.json"
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith("run_")):
        result_path = run_dir / file_name
        rows = _safe_load_json(result_path)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                yield run_dir.name, row


def analyze_eval_dir(eval_dir: Path, variant: str, examples: int) -> str:
    pred_counts: Counter[str] = Counter()
    raw_extracted_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    rows_total = 0
    llm_error_count = 0
    llm_warning_count = 0
    llm_missing_prediction_count = 0
    pred_missing_count = 0
    llm_processed_present = 0
    repair_used_count = 0
    parse_disagree_count = 0
    parse_disagree_to_servant1_count = 0
    role_order_in_prompt_count = 0
    prompt_count = 0

    correct_original = 0
    total_original = 0
    correct_merged = 0
    total_merged = 0

    servant1_examples: List[Tuple[str, str, str, str, str]] = []

    for run_name, row in _iter_result_rows(eval_dir, variant):
        rows_total += 1
        cls = row.get("classification") or {}
        log_row = row.get("log_row") or {}

        true_role = cls.get("true_role") if isinstance(cls.get("true_role"), str) else None
        pred_role = cls.get("pred_role") if isinstance(cls.get("pred_role"), str) else None

        if true_role:
            true_counts[true_role] += 1
        if pred_role:
            pred_counts[pred_role] += 1
        else:
            pred_missing_count += 1

        if true_role and pred_role:
            confusion[true_role][pred_role] += 1
            total_original += 1
            if true_role == pred_role:
                correct_original += 1

            merged_true = _map_merged_servant(true_role)
            merged_pred = _map_merged_servant(pred_role)
            total_merged += 1
            if merged_true == merged_pred:
                correct_merged += 1

        llm_error = log_row.get("llm_error")
        llm_warning = log_row.get("llm_warning")
        if llm_error not in (None, ""):
            llm_error_count += 1
        if llm_warning not in (None, ""):
            llm_warning_count += 1

        llm_prediction = (row.get("llm") or {}).get("prediction")
        if not isinstance(llm_prediction, str) or not llm_prediction.strip():
            llm_missing_prediction_count += 1
            raw_role = None
        else:
            raw_role = _extract_role_like_pipeline(llm_prediction)
            if raw_role:
                raw_extracted_counts[raw_role] += 1

        llm_processed = row.get("llm_processed")
        if isinstance(llm_processed, dict):
            llm_processed_present += 1
            if bool(llm_processed.get("used_repair")):
                repair_used_count += 1

        if raw_role and pred_role and raw_role != pred_role:
            parse_disagree_count += 1
            if pred_role == "servant-1":
                parse_disagree_to_servant1_count += 1

        llm_io = row.get("llm_io") or {}
        prompt_candidate = llm_io.get("prompt_used") or llm_io.get(f"prompt_{variant}")
        if isinstance(prompt_candidate, str):
            prompt_count += 1
            if "Valid roles: assassin, merlin, morgana, percival, servant-1, servant-2" in prompt_candidate:
                role_order_in_prompt_count += 1

        if pred_role == "servant-1" and len(servant1_examples) < examples:
            target = row.get("target") or {}
            entry_id = str(target.get("entry_id", ""))
            raw_preview = (llm_prediction or "").strip().replace("\n", " ")[:160]
            servant1_examples.append(
                (
                    run_name,
                    entry_id,
                    str(true_role),
                    str(raw_role),
                    raw_preview,
                )
            )

    lines: List[str] = []
    lines.append(f"=== {eval_dir} | {variant} ===")
    lines.append(f"Rows analyzed: {rows_total}")
    if rows_total == 0:
        lines.append("No rows found.")
        return "\n".join(lines)

    lines.append("\nPred distribution (final classification.pred_role):")
    for role_name, count in pred_counts.most_common():
        lines.append(f"- {role_name:10s} : {count:4d} ({_pct(count, rows_total):5.1f}%)")

    lines.append("\nRaw distribution (role parsed from llm.prediction text):")
    for role_name, count in raw_extracted_counts.most_common():
        lines.append(f"- {role_name:10s} : {count:4d} ({_pct(count, rows_total):5.1f}%)")

    servant_1 = pred_counts.get("servant-1", 0)
    servant_2 = pred_counts.get("servant-2", 0)
    skew_ratio = (servant_1 / servant_2) if servant_2 > 0 else float("inf")
    lines.append(
        f"\nServant skew (final): servant-1={servant_1}, servant-2={servant_2}, "
        f"ratio={skew_ratio:.2f}" if servant_2 > 0 else "\nServant skew (final): servant-2 never predicted"
    )

    s2_total = true_counts.get("servant-2", 0)
    s1_total = true_counts.get("servant-1", 0)
    s2_to_s1 = confusion["servant-2"].get("servant-1", 0)
    s1_to_s2 = confusion["servant-1"].get("servant-2", 0)
    lines.append(
        f"True servant-2 -> pred servant-1: {s2_to_s1}/{s2_total} ({_pct(s2_to_s1, s2_total):5.1f}%)"
    )
    lines.append(
        f"True servant-1 -> pred servant-2: {s1_to_s2}/{s1_total} ({_pct(s1_to_s2, s1_total):5.1f}%)"
    )

    lines.append(
        f"\nLLM diagnostics: errors={llm_error_count}, warnings={llm_warning_count}, "
        f"missing_llm_prediction={llm_missing_prediction_count}, missing_pred_role={pred_missing_count}"
    )
    lines.append(
        f"LLM post-processor records: llm_processed_present={llm_processed_present}, used_repair={repair_used_count}"
    )
    lines.append(
        f"Raw-vs-final parse disagreements: {parse_disagree_count} "
        f"(to servant-1: {parse_disagree_to_servant1_count})"
    )
    lines.append(
        f"Prompt contains fixed role-order string: {role_order_in_prompt_count}/{prompt_count} "
        f"({_pct(role_order_in_prompt_count, prompt_count):5.1f}%)"
    )

    if total_original > 0:
        orig_acc = _pct(correct_original, total_original)
        merged_acc = _pct(correct_merged, total_merged)
        lines.append(
            f"\nAccuracy (original labels): {correct_original}/{total_original} ({orig_acc:5.1f}%)"
        )
        lines.append(
            f"Accuracy (merged servant labels): {correct_merged}/{total_merged} ({merged_acc:5.1f}%)"
        )
        lines.append(f"Merged-servant gain: {merged_acc - orig_acc:+.1f} pp")

    lines.append("\nTop confusion rows (selected true roles):")
    for true_role in ["servant-1", "servant-2", "merlin", "morgana", "percival", "assassin"]:
        if true_counts[true_role] == 0:
            continue
        top_preds = confusion[true_role].most_common(3)
        rendered = ", ".join(f"{pred}:{count}" for pred, count in top_preds)
        lines.append(f"- true={true_role:10s} n={true_counts[true_role]:3d} -> {rendered}")

    if servant1_examples:
        lines.append("\nExample final servant-1 cases:")
        for run_name, entry_id, true_role, raw_role, raw_preview in servant1_examples:
            lines.append(
                f"- {run_name:6s} | {entry_id:18s} | true={true_role:10s} | raw_role={raw_role:10s} | {raw_preview}"
            )

    return "\n".join(lines)


def compare_variants(eval_dir: Path, examples: int) -> str:
    baseline_report = analyze_eval_dir(eval_dir, "baseline", examples)
    current_report = analyze_eval_dir(eval_dir, "current", examples)
    return f"{baseline_report}\n\n{'-' * 100}\n\n{current_report}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify servant-1 bias claims in evaluation outputs using direct counts."
    )
    parser.add_argument(
        "eval_dirs",
        nargs="+",
        type=Path,
        help="One or more eval experiment directories (must contain runs/run_*/results_*.json).",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "current", "both"],
        default="both",
        help="Which results file variant to analyze.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=8,
        help="How many servant-1 example rows to print for each section.",
    )
    args = parser.parse_args()

    for index, eval_dir in enumerate(args.eval_dirs):
        resolved = eval_dir.resolve()
        if args.variant == "both":
            print(compare_variants(resolved, args.examples))
        else:
            print(analyze_eval_dir(resolved, args.variant, args.examples))
        if index < len(args.eval_dirs) - 1:
            print("\n" + "=" * 120 + "\n")


if __name__ == "__main__":
    main()
