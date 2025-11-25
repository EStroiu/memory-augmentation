#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Protocol, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
from scripts.metrics_utils import average_aggregates, viz_prompt_lengths, viz_confusion_matrix
from scripts.template_summary import (
    MemoryEntry,
    load_game_json,
    build_memory_entries,
    build_embeddings,
    build_faiss_ip_index,
    retrieve_top,
    messages_by_quest,
    quest_state_line,
    quest_transcript_only,
)
from scripts.llm_client import llm_role_predict
from scripts.json_utils import (
    extract_beliefs_object,
    extract_role_label,
    typechat_repair_to_json,
)
from scripts.prompting import (
    assemble_prompt_with_meta,
    assemble_baseline_prompt_with_meta,
    assemble_belief_baseline_prompt,
    prompt_stats,
)

# Ensure repo root is importable when running as a script (python scripts/...)
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class ExperimentConfig:
    """High-level configuration for a single experiment run.

    This can be used both from the CLI and from notebooks/other Python code
    to launch experiments with a single function call.
    """

    data_dir: Path
    outdir: Path = Path("outputs/eval")
    k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_llm: bool = False
    llm_model: str = "ollama:llama2:13b"
    openai_api_key: Optional[str] = None
    max_games: Optional[int] = None
    num_runs: int = 1
    seed: int = 42

    # Prompt strategies (registered in PROMPT_REGISTRY)
    baseline_prompt: str = "baseline_full_transcript"  # or "belief_vector", "none"
    current_prompt: str = "mem_template"               # or "mem_template+summary", ...
    run_baseline: bool = True
    run_current: bool = True

    # LLM post-processing (registered in LLM_POST_REGISTRY)
    llm_fixer: str = "none"  # e.g. "none", "typechat_role", "typechat_beliefs"

    # Human-readable name for output folder naming
    experiment_name: str = "custom"


class PromptStrategy(Protocol):
    def __call__(
        self,
        target: MemoryEntry,
        retrieved: List[Tuple[MemoryEntry, float]],
        ctx: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        ...


class LLMPostProcessor(Protocol):
    def __call__(
        self,
        raw_pred: Optional[str],
        valid_roles: List[str],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


PROMPT_REGISTRY: Dict[str, PromptStrategy] = {}
LLM_POST_REGISTRY: Dict[str, LLMPostProcessor] = {}


def register_prompt(name: str) -> Callable[[PromptStrategy], PromptStrategy]:
    def _wrap(fn: PromptStrategy) -> PromptStrategy:
        PROMPT_REGISTRY[name] = fn
        return fn

    return _wrap


def register_llm_post(name: str) -> Callable[[LLMPostProcessor], LLMPostProcessor]:
    def _wrap(fn: LLMPostProcessor) -> LLMPostProcessor:
        LLM_POST_REGISTRY[name] = fn
        return fn

    return _wrap


@register_prompt("baseline_full_transcript")
def prompt_baseline_full_transcript(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    task_suffix: str = ctx.get("task_suffix", "")
    prompt, meta = assemble_baseline_prompt_with_meta(target, task_suffix=task_suffix)
    meta["mode"] = "full_transcript"
    return prompt, meta


@register_prompt("belief_vector")
def prompt_belief_vector(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Belief-vector baseline prompt using cached game structures in ctx."""

    game_id: str = target.game_id
    quests: Dict[int, List[Dict]] = ctx["game_quests"].get(game_id, {})
    players: List[str] = ctx["game_players"].get(game_id, [])
    belief_state_by_game: Dict[str, Dict[str, str]] = ctx["belief_state_by_game"]
    valid_roles: List[str] = ctx["valid_roles"]

    # Ensure stable belief state dict for this game
    beliefs = belief_state_by_game[game_id]
    for p in players:
        beliefs.setdefault(p, "unknown")

    # Past state lines for quests < current quest
    past_state_lines: List[str] = []
    for q in sorted(quests.keys()):
        if int(q) >= int(target.quest):
            continue
        past_state_lines.append(quest_state_line(game_id, int(q), quests[q]))
    current_msgs = quests.get(int(target.quest), [])
    current_transcript = quest_transcript_only(current_msgs)

    prompt = assemble_belief_baseline_prompt(
        game_id=game_id,
        players=players,
        belief_vector=beliefs,
        past_state_lines=past_state_lines,
        current_quest=int(target.quest),
        current_transcript=current_transcript,
        valid_roles=valid_roles,
    )
    meta: Dict[str, Any] = {"mode": "belief_vector"}
    return prompt, meta


@register_prompt("mem_template")
def prompt_mem_template(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    task_suffix: str = ctx.get("task_suffix", "")
    return assemble_prompt_with_meta(target, retrieved, task_suffix=task_suffix)


@register_prompt("mem_template+summary")
def prompt_mem_template_summary(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Prompt strategy for template+summary memory format.

    This assumes entries were built with memory_format="template+summary", which
    appends a heuristic summary to each template entry in template_summary.
    """
    task_suffix: str = ctx.get("task_suffix", "")
    return assemble_prompt_with_meta(target, retrieved, task_suffix=task_suffix)


@register_llm_post("none")
def llm_post_none(
    raw_pred: Optional[str],
    valid_roles: List[str],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """No repair: just parse JSON/labels locally."""

    pred_role = extract_role_label(raw_pred, valid_roles)
    return {
        "pred_role": pred_role,
        "beliefs_obj": None,
        "used_repair": False,
        "proposer_from_beliefs": None,
        "beliefs_preview": None,
    }


@register_llm_post("typechat_role")
def llm_post_typechat_role(
    raw_pred: Optional[str],
    valid_roles: List[str],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """TypeChat-style repair for role-only JSON."""

    openai_model: str = ctx.get("openai_model", "")
    openai_api_key: Optional[str] = ctx.get("openai_api_key")
    base_role = extract_role_label(raw_pred, valid_roles)
    used_repair = False
    if base_role is None and raw_pred:
        schema_hint_role = '{"role": "<ROLE>"}'
        repaired_role = typechat_repair_to_json(raw_pred, schema_hint_role, openai_model, openai_api_key, True)
        repaired_parsed = extract_role_label(repaired_role, valid_roles) if repaired_role else None
        if repaired_parsed is not None:
            base_role = repaired_parsed
            used_repair = True
    return {
        "pred_role": base_role,
        "beliefs_obj": None,
        "used_repair": used_repair,
        "proposer_from_beliefs": None,
        "beliefs_preview": None,
    }


@register_llm_post("typechat_beliefs")
def llm_post_typechat_beliefs(
    raw_pred: Optional[str],
    valid_roles: List[str],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Belief-vector style repair: parse/update beliefs and derive proposer role."""

    openai_model: str = ctx.get("openai_model", "")
    openai_api_key: Optional[str] = ctx.get("openai_api_key")
    target: MemoryEntry = ctx["target"]
    belief_state_by_game: Dict[str, Dict[str, str]] = ctx["belief_state_by_game"]

    beliefs_obj: Dict[str, Any] | None = None
    used_repair = False
    beliefs_preview: Optional[str] = None
    proposer_from_beliefs: Optional[str] = None

    text = raw_pred or ""
    beliefs_container = extract_beliefs_object(text)
    if isinstance(beliefs_container, dict):
        beliefs_obj = beliefs_container.get("beliefs") if isinstance(beliefs_container.get("beliefs"), dict) else None
    if beliefs_obj is None and text:
        schema_hint = '{"beliefs": {"player-1": "<ROLE>", "player-2": "<ROLE>", ...}}'
        repaired = typechat_repair_to_json(text, schema_hint, openai_model, openai_api_key, True)
        repaired_container = extract_beliefs_object(repaired or "") if repaired else None
        if isinstance(repaired_container, dict):
            beliefs_obj = repaired_container.get("beliefs") if isinstance(repaired_container.get("beliefs"), dict) else None
            used_repair = beliefs_obj is not None

    pred_role = extract_role_label(raw_pred, valid_roles)
    if isinstance(beliefs_obj, dict):
        beliefs = belief_state_by_game[target.game_id]
        for p, r in beliefs_obj.items():
            if isinstance(r, str):
                beliefs[str(p)] = r.strip()
        try:
            beliefs_preview = json.dumps({"beliefs": beliefs_obj})[:160]
        except Exception:
            beliefs_preview = None
        if isinstance(target.proposer, str):
            pr = beliefs.get(target.proposer)
            if isinstance(pr, str) and any(pr.lower() == vr.lower() for vr in valid_roles):
                pred_role = next(vr for vr in valid_roles if pr.lower() == vr.lower())
                proposer_from_beliefs = pred_role

    # If still no pred_role, fallback to role repair
    if pred_role is None and text:
        schema_hint_role = '{"role": "<ROLE>"}'
        repaired_role = typechat_repair_to_json(text, schema_hint_role, openai_model, openai_api_key, True)
        pr2 = extract_role_label(repaired_role, valid_roles) if repaired_role else None
        if pr2 is not None and pred_role is None:
            used_repair = True
        pred_role = pred_role or pr2

    return {
        "pred_role": pred_role,
        "beliefs_obj": beliefs_obj,
        "used_repair": used_repair,
        "proposer_from_beliefs": proposer_from_beliefs,
        "beliefs_preview": beliefs_preview,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate_config(
    files: List[Path],
    k: int,
    memory_format: str,
    model_name: str,
    use_llm: bool,
    openai_model: str,
    openai_api_key: str | None,
    llm_use_baseline_prompt: bool,
    baseline_mode: str = "full_transcript",
    prompt_name: str = "mem_template",
    llm_fixer: str = "none",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one configuration and return (aggregate, results).

    Retrieval metrics exclude the query (self) from candidate results to avoid trivial rank=1.
    Positives are defined as entries from the same game as the query (excluding self).
    """
    t0 = time.time()
    # Build entries across all games
    all_entries: List[MemoryEntry] = []
    per_game_indices: Dict[str, List[int]] = defaultdict(list)
    # For belief baseline: cache game-level structures
    games_raw: Dict[str, Dict] = {}
    game_quests: Dict[str, Dict[int, List[Dict]]] = {}
    game_players: Dict[str, List[str]] = {}
    for f in files:
        game = load_game_json(f)
        gid = f.stem
        games_raw[gid] = game
        quests = messages_by_quest(game)
        game_quests[gid] = quests
        # players list (stable order) from users map if available
        users = (game.get("users", {}) or {})
        players = [u.get("name") for u in users.values() if u.get("name")]
        game_players[gid] = [str(p) for p in players]
        entries = build_memory_entries(game, gid, memory_format=memory_format)
        start_idx = len(all_entries)
        all_entries.extend(entries)
        per_game_indices[gid].extend(list(range(start_idx, start_idx + len(entries))))
    build_elapsed = time.time() - t0
    print(f"    [progress] Built {len(all_entries)} entries (memory_format='{memory_format}') in {build_elapsed:.2f}s")

    # Embeddings and index
    t1 = time.time()
    emb, model = build_embeddings(all_entries, model_name)
    index = build_faiss_ip_index(emb)
    embed_elapsed = time.time() - t1
    print(f"    [progress] Embeddings + index ready in {embed_elapsed:.2f}s (model='{model_name}')")

    # Enumerate valid roles from dataset once (for parsing LLM output consistently)
    valid_roles: List[str] = sorted({e.proposer_role for e in all_entries if e.proposer_role})  # type: ignore[arg-type]

    results: List[Dict[str, Any]] = []
    # Classification counters for proposer-role prediction via retrieval (top-1 label transfer)
    roles_seen: set[str] = set()
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    # Confusion matrix counts: true_role -> pred_role -> count
    conf_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    loop_start = time.time()
    total_entries = len(all_entries)
    last_report = 0.0

    # For belief baseline and LLM post-processing: per-game belief vectors as dict[player] -> role
    belief_state_by_game: Dict[str, Dict[str, str]] = defaultdict(dict)

    for qi, target in enumerate(all_entries):
        q_vec = model.encode([target.text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        # Retrieve more than k for stability
        D, I = retrieve_top(index, q_vec, topn=min(max(50, k * 5), len(all_entries)))

        # Exclude self from candidate pool
        mask = I != qi
        I = I[mask]
        D = D[mask]

        # Use nearest-neighbor scores directly for memory augmentation
        I2, D2 = I, D

        # Top-k selection
        topk_idx = list(I2[: k])
        topk_scores = list(D2[: k])

        # Prompts and stats (strategy-based)
        retrieved_pairs = [(all_entries[j], float(s)) for j, s in zip(topk_idx, topk_scores)]
        roles_list = ", ".join(valid_roles) if valid_roles else ""
        task_suffix = "\n\nTask: Predict the role of the player who proposed the party."
        if roles_list:
            task_suffix += f" Valid roles: {roles_list}."
        task_suffix += (
            " Respond ONLY with a single-line JSON object matching this schema: "
            '{"role": "<ROLE>"}. Do not include explanations, prefixes, or extra keys.'
        )

        ctx: Dict[str, Any] = {
            "task_suffix": task_suffix,
            "game_quests": game_quests,
            "game_players": game_players,
            "belief_state_by_game": belief_state_by_game,
            "valid_roles": valid_roles,
        }

        prompt_fn = PROMPT_REGISTRY.get(prompt_name)
        if prompt_fn is None:
            raise ValueError(f"Unknown prompt strategy: {prompt_name}")

        llm_prompt_text, prompt_meta = prompt_fn(target, retrieved_pairs, ctx)

        # For backwards-compatible stats, treat this single prompt as both baseline and with-memory.
        prompt_base_task = llm_prompt_text
        prompt_mem_task = llm_prompt_text
        base_meta = prompt_meta
        mem_meta = prompt_meta

        pstats = prompt_stats(prompt_mem_task, prompt_base_task)

        # Optional LLM call (uses selected prompt depending on run mode)
        llm_prompt_to_use = prompt_base_task if bool(use_llm) and bool(llm_use_baseline_prompt) else prompt_mem_task
        llm_out = llm_role_predict(llm_prompt_to_use, bool(use_llm), openai_model, openai_api_key)

        # Lightweight logging of LLM response and prompt length to monitor truncation/complaints
        if bool(use_llm):
            preview_raw = (llm_out.get("prediction") or "")
            preview = preview_raw.replace("\n", " ")[:160]
            err = llm_out.get("error") or llm_out.get("note")
            warn = llm_out.get("warning") or llm_out.get("truncated")
            # Clearer, aligned multi-line log for readability
            print(
                """
    [llm]
      game: {game}  quest: {quest}
      prompt_chars: {pchars}
      error: {error}
      warning: {warning}
      prediction_preview: {preview}
                """.strip().format(
                    game=gid,
                    quest=target.quest,
                    pchars=len(llm_prompt_to_use),
                    error=repr(err),
                    warning=repr(warn),
                    preview=repr(preview),
                )
            )

        # Proposer-role prediction (LLM or heuristic fallback)
        true_role = target.proposer_role
        pred_role = None
        processed_used_repair: bool = False
        processed_beliefs_preview: str | None = None
        proposer_from_beliefs: str | None = None
        if bool(use_llm):
            raw_pred = llm_out.get("prediction")
            post_fn = LLM_POST_REGISTRY.get(llm_fixer)
            if post_fn is None:
                raise ValueError(f"Unknown LLM post-processor: {llm_fixer}")
            post_out = post_fn(
                raw_pred,
                valid_roles,
                {
                    "openai_model": openai_model,
                    "openai_api_key": openai_api_key,
                    "target": target,
                    "belief_state_by_game": belief_state_by_game,
                },
            )
            pred_role = post_out.get("pred_role")
            processed_used_repair = bool(post_out.get("used_repair"))
            processed_beliefs_preview = post_out.get("beliefs_preview")
            proposer_from_beliefs = post_out.get("proposer_from_beliefs")
        else:
            # Fallback heuristic: transfer top-1 retrieved proposer_role
            if len(I2) > 0:
                cand_role = all_entries[int(I2[0])].proposer_role
                if cand_role is not None:
                    pred_role = cand_role
        if true_role is not None:
            roles_seen.add(true_role)
        if pred_role is not None:
            roles_seen.add(pred_role)
        # Update confusion counts when both labels known
        if true_role is not None and pred_role is not None:
            if pred_role == true_role:
                tp[true_role] += 1
            else:
                fp[pred_role] += 1
                fn[true_role] += 1
            conf_counts[true_role][pred_role] += 1

        # Optional: print a processed view for debugging
        if bool(use_llm):
            print(
                (
                    """
    [llm-processed]
      game: {game}  quest: {quest}
      role_parsed: {role}
      proposer_from_beliefs: {proposer_role}
      used_repair: {used_repair}
      beliefs_preview: {beliefs_preview}
                    """.strip()
                ).format(
                    game=gid,
                    quest=target.quest,
                    role=repr(pred_role),
                    proposer_role=repr(proposer_from_beliefs),
                    used_repair=repr(processed_used_repair),
                    beliefs_preview=repr(processed_beliefs_preview),
                )
            )

        res = {
            "target": asdict(target),
            "topk": [
                {"entry_id": all_entries[j].entry_id, "game_id": all_entries[j].game_id, "quest": int(all_entries[j].quest), "score": float(s)}
                for j, s in zip(topk_idx, topk_scores)
            ],
            "memory_format": memory_format,
            "prompt_stats": pstats,
            "prompt_meta": {"with_memory": mem_meta, "baseline": base_meta},
            "llm": llm_out,
            "classification": {
                "true_role": true_role,
                "pred_role": pred_role,
            },
        }
        results.append(res)

        # Progress print every ~10% or each 100 entries
        if total_entries > 0:
            pct = (qi + 1) / total_entries * 100.0
            if pct - last_report >= 10.0 or (qi + 1) % 100 == 0 or (qi + 1) == total_entries:
                last_report = pct
                elapsed_loop = time.time() - loop_start
                avg_per = elapsed_loop / (qi + 1)
                est_total = avg_per * total_entries
                remaining = est_total - elapsed_loop
                print(f"    [progress] {qi+1}/{total_entries} ({pct:4.1f}%) entries; avg {avg_per:.2f}s/entry; ETA {remaining:.1f}s")

    # Aggregations
    avg_prompt_sizes = {
        "with_memory": {
            "chars": float(np.mean([r["prompt_stats"]["with_memory"]["chars"] for r in results])) if results else 0.0,
            "words": float(np.mean([r["prompt_stats"]["with_memory"]["words"] for r in results])) if results else 0.0,
            "lines": float(np.mean([r["prompt_stats"]["with_memory"]["lines"] for r in results])) if results else 0.0,
        },
        "baseline": {
            "chars": float(np.mean([r["prompt_stats"]["baseline"]["chars"] for r in results])) if results else 0.0,
            "words": float(np.mean([r["prompt_stats"]["baseline"]["words"] for r in results])) if results else 0.0,
            "lines": float(np.mean([r["prompt_stats"]["baseline"]["lines"] for r in results])) if results else 0.0,
        },
    }

    # Classification metrics (per-role F1 and micro-F1)
    roles_sorted = sorted(roles_seen)
    clf_by_role: Dict[str, Dict[str, float]] = {}
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for r in roles_sorted:
        t = float(tp.get(r, 0))
        f_p = float(fp.get(r, 0))
        f_n = float(fn.get(r, 0))
        tp_sum += int(t)
        fp_sum += int(f_p)
        fn_sum += int(f_n)
        prec = (t / (t + f_p)) if (t + f_p) > 0 else 0.0
        rec = (t / (t + f_n)) if (t + f_n) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        clf_by_role[r] = {"precision": prec, "recall": rec, "f1": f1, "tp": t, "fp": f_p, "fn": f_n}
    micro_prec = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_rec = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    total_elapsed = time.time() - t0
    print(f"    [done] memory_format='{memory_format}' finished in {total_elapsed:.2f}s")
    agg = {
        "k": int(k),
        "memory_format": memory_format,
        "model": model_name,
        "avg_prompt_sizes": avg_prompt_sizes,
        "clf_by_role": clf_by_role,
        "clf_micro_f1": micro_f1,
        "clf_confusion": {tr: dict(prs) for tr, prs in conf_counts.items()},
    }
    return agg, results


# ---------- Combined evaluation (baseline vs current) ----------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval and prompting pipeline over an Avalon dataset.")
    ap.add_argument("--data_dir", type=str, default="dataset", help="Directory containing *.json games")
    ap.add_argument("--k", type=int, default=3, help="Top-k for retrieval")
    ap.add_argument("--memory_format", type=str, default="template", choices=["template", "template+summary"], help="Memory format for entries")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer embedding model name")
    ap.add_argument("--outdir", type=str, default="outputs/eval", help="Base output directory for saving results and visuals")
    ap.add_argument("--llm", action="store_true", help="Call an LLM for downstream role prediction (TypeChat-like JSON output enforced)")
    ap.add_argument("--llm_model", type=str, default="ollama:llama2:13b", help="LLM identifier. Use 'ollama:<model>' (e.g., 'ollama:llama3:8b-instruct') or 'openai:<model>' for OpenAI.")
    ap.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key (if --llm)")
    ap.add_argument("--max_games", type=int, default=None, help="Limit number of games for quick tests")
    ap.add_argument("--num_runs", type=int, default=1, help="Repeat the experiment N times and average results")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed (incremented per run)")

    # New high-level experiment knobs
    ap.add_argument("--baseline_prompt", type=str, default="baseline_full_transcript", help="Prompt strategy name for baseline (see README for options). Use 'none' to skip.")
    ap.add_argument("--current_prompt", type=str, default="mem_template", help="Prompt strategy name for current (memory-augmented) run. Use 'none' to skip.")
    ap.add_argument("--llm_fixer", type=str, default="none", choices=list(LLM_POST_REGISTRY.keys()), help="How to post-process LLM JSON output (TypeChat repair, etc.).")
    ap.add_argument("--exp", type=str, default="custom", choices=["custom", "baseline_full", "baseline_vs_template", "baseline_vs_template+summary"], help="Named experiment preset. 'custom' uses provided arguments.")
    args = ap.parse_args(argv)

    # Heuristic auto-correction: users sometimes pass the LLM model to --model.
    # If --model looks like an LLM identifier and --llm_model is still default, swap them.
    try:
        default_llm_model = ap.get_default("llm_model")
        default_embed_model = ap.get_default("model")
        if args.llm and isinstance(args.model, str) and args.model.lower().startswith(("ollama:", "openai:")):
            # Only swap if user did not explicitly set --llm_model different from default.
            if args.llm_model == default_llm_model:
                print(
                    f"[warn] Detected LLM id '{args.model}' passed to --model. "
                    f"Treating it as --llm_model and restoring embedding model to '{default_embed_model}'.",
                    file=sys.stderr,
                )
                args.llm_model = args.model
                args.model = default_embed_model
    except Exception:
        pass

    # Apply experiment presets on top of parsed args
    if args.exp == "baseline_full":
        args.baseline_prompt = "baseline_full_transcript"
        args.current_prompt = "none"
    elif args.exp == "baseline_vs_template":
        args.baseline_prompt = "baseline_full_transcript"
        args.current_prompt = "mem_template"
    elif args.exp == "baseline_vs_template+summary":
        args.baseline_prompt = "baseline_full_transcript"
        args.current_prompt = "mem_template+summary"

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        return 1

    files = sorted(data_dir.glob("*.json"))
    if args.max_games:
        files = files[: int(args.max_games)]
    if not files:
        print(f"No JSON games in {data_dir}", file=sys.stderr)
        return 1

    # Pre-run workload estimation (entries per memory format, projected LLM calls & duration)
    try:
        baseline_entries_count = 0
        current_entries_count = 0
        for f in files:
            game_obj = load_game_json(f)
            baseline_entries_count += len(build_memory_entries(game_obj, f.stem, memory_format="template"))
            current_entries_count += len(build_memory_entries(game_obj, f.stem, memory_format=args.memory_format))
        total_llm_calls_per_run = (baseline_entries_count + current_entries_count) if args.llm else 0
        total_llm_calls_all = total_llm_calls_per_run * int(args.num_runs)
        avg_latency = float(os.getenv("LLM_AVG_LATENCY_SEC", "2.5")) if args.llm else 0.0
        est_minutes = (total_llm_calls_all * avg_latency) / 60.0 if avg_latency > 0 else 0.0
        print(f"[info] Loaded {len(files)} game files from '{data_dir}'.")
        print(f"[info] Baseline entries (template): {baseline_entries_count}")
        print(f"[info] Current entries ({args.memory_format}): {current_entries_count}")
        if args.llm:
            print(f"[info] Planned runs: {args.num_runs}; LLM calls/run: {total_llm_calls_per_run}; total: {total_llm_calls_all}")
            print(f"[estimate] Avg latency ~{avg_latency:.1f}s -> est prompting time ~{est_minutes:.1f} min (override with LLM_AVG_LATENCY_SEC)")
        else:
            print("[info] LLM disabled; using retrieval heuristic for role prediction.")
    except Exception as e:
        print(f"[warn] Pre-run estimation unavailable: {e}")

    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = Path(args.outdir) / f"{ts}_k-{args.k}_mem-{args.memory_format}_exp-{args.exp}"
    ensure_dir(run_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: getattr(args, k) for k in vars(args)}, f, indent=2)

    # Multi-run loop: save per-run artifacts in subfolders and average aggregates
    agg_base_runs: List[Dict[str, Any]] = []
    agg_cur_runs: List[Dict[str, Any]] = []
    for i in range(int(args.num_runs)):
        print(f"Run {i+1}/{args.num_runs}")
        run_subdir = run_dir / f"runs/run_{i+1}"
        ensure_dir(run_subdir)
        # Optional: set seeds (might not affect deterministic parts but harmless)
        try:
            np.random.seed(int(args.seed) + i)
        except Exception:
            pass

        # Baseline
        if args.baseline_prompt != "none":
            print(f"  Evaluating baseline strategy='{args.baseline_prompt}', memory_format=template")
            agg_base_i, res_base_i = evaluate_config(
                files=files,
                k=int(args.k),
                memory_format="template",
                model_name=args.model,
                use_llm=bool(args.llm),
                openai_model=args.llm_model,
                openai_api_key=args.openai_api_key,
                llm_use_baseline_prompt=True,
                baseline_mode="full_transcript",
                prompt_name=args.baseline_prompt,
                llm_fixer=args.llm_fixer,
            )
            agg_base_runs.append(agg_base_i)
            with (run_subdir / "results_baseline.json").open("w", encoding="utf-8") as f:
                json.dump(res_base_i, f, ensure_ascii=False, indent=2)
            with (run_subdir / "aggregate_baseline.json").open("w", encoding="utf-8") as f:
                json.dump(agg_base_i, f, ensure_ascii=False, indent=2)

        # Current
        if args.current_prompt != "none":
            print(f"  Evaluating current strategy='{args.current_prompt}', memory_format={args.memory_format}")
            agg_cur_i, res_cur_i = evaluate_config(
                files=files,
                k=int(args.k),
                memory_format=args.memory_format,
                model_name=args.model,
                use_llm=bool(args.llm),
                openai_model=args.llm_model,
                openai_api_key=args.openai_api_key,
                llm_use_baseline_prompt=False,
                baseline_mode="full_transcript",
                prompt_name=args.current_prompt,
                llm_fixer=args.llm_fixer,
            )
            agg_cur_runs.append(agg_cur_i)
            with (run_subdir / "results_current.json").open("w", encoding="utf-8") as f:
                json.dump(res_cur_i, f, ensure_ascii=False, indent=2)
            with (run_subdir / "aggregate_current.json").open("w", encoding="utf-8") as f:
                json.dump(agg_cur_i, f, ensure_ascii=False, indent=2)

    # Averages across runs
    agg_base = average_aggregates(agg_base_runs) if agg_base_runs else {}
    agg_cur = average_aggregates(agg_cur_runs) if agg_cur_runs else {}
    if agg_base_runs:
        with (run_dir / "aggregate_baseline_mean.json").open("w", encoding="utf-8") as f:
            json.dump(agg_base, f, ensure_ascii=False, indent=2)
    if agg_cur_runs:
        with (run_dir / "aggregate_current_mean.json").open("w", encoding="utf-8") as f:
            json.dump(agg_cur, f, ensure_ascii=False, indent=2)
    combined = {"baseline": agg_base, "current": agg_cur}
    with (run_dir / "aggregate_combined_mean.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # Write F1 results as CSV (and keep txt for convenience)
    roles = sorted(set(list(agg_base.get("clf_by_role", {}).keys()) + list(agg_cur.get("clf_by_role", {}).keys())))
    if roles:
        # CSV
        import csv
        with (run_dir / "role_f1_table.csv").open("w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["approach", *roles, "micro_f1"])
            writer.writerow(["baseline", *[f"{agg_base['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles], f"{agg_base.get('clf_micro_f1', 0.0):.3f}"])
            writer.writerow(["current", *[f"{agg_cur['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles], f"{agg_cur.get('clf_micro_f1', 0.0):.3f}"])
        # TXT (tab-delimited)
        header = ["approach"] + roles + ["micro_f1"]
        lines = ["\t".join(header)]
        base_vals = ["baseline"] + [f"{agg_base['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles] + [f"{agg_base.get('clf_micro_f1', 0.0):.3f}"]
        cur_vals = ["current"] + [f"{agg_cur['clf_by_role'].get(r, {}).get('f1', 0.0):.3f}" for r in roles] + [f"{agg_cur.get('clf_micro_f1', 0.0):.3f}"]
        (run_dir / "role_f1_table.txt").write_text("\n".join([*lines, "\t".join(base_vals), "\t".join(cur_vals)]) + "\n", encoding="utf-8")

        # Optional: save per-run F1s for transparency (if num_runs > 1)
        if int(getattr(args, "num_runs", 1)) > 1:
            with (run_dir / "role_f1_table_runs.csv").open("w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["run", "approach", "role", "f1"])
                # baseline
                for i, a in enumerate(agg_base_runs, start=1):
                    for r in roles:
                        writer.writerow([i, "baseline", r, f"{a.get('clf_by_role', {}).get(r, {}).get('f1', 0.0):.3f}"])
                # current
                for i, a in enumerate(agg_cur_runs, start=1):
                    for r in roles:
                        writer.writerow([i, "current", r, f"{a.get('clf_by_role', {}).get(r, {}).get('f1', 0.0):.3f}"])

    # Visualizations: confusion matrices and current prompt sizes
    viz_prompt_lengths(run_dir, agg_cur["avg_prompt_sizes"])  # focuses on current config’s prompts

    # Note: Retrieval-only plots intentionally omitted in LLM-only mode.

    # New: Confusion matrices by role (averaged across runs)
    cm_roles = sorted(set(list(agg_base.get("clf_by_role", {}).keys()) + list(agg_cur.get("clf_by_role", {}).keys())))
    if cm_roles:
        # Baseline
        viz_confusion_matrix(
            run_dir,
            title="Confusion matrix (baseline)",
            roles=cm_roles,
            confusion=agg_base.get("clf_confusion", {}),
            f1_by_role={r: agg_base.get("clf_by_role", {}).get(r, {}).get("f1", 0.0) for r in cm_roles},
            filename="confusion_baseline.html",
        )
        # Current
        viz_confusion_matrix(
            run_dir,
            title="Confusion matrix (current)",
            roles=cm_roles,
            confusion=agg_cur.get("clf_confusion", {}),
            f1_by_role={r: agg_cur.get("clf_by_role", {}).get(r, {}).get("f1", 0.0) for r in cm_roles},
            filename="confusion_current.html",
        )

    print(f"Saved evaluation to: {run_dir}")
    print(
        f"Baseline (mean of {args.num_runs} runs): micro-F1={agg_base.get('clf_micro_f1', 0.0):.3f} | "
        f"Current (mean of {args.num_runs} runs): micro-F1={agg_cur.get('clf_micro_f1', 0.0):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
