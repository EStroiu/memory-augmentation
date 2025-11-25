#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
from scripts.metrics_utils import average_aggregates, viz_prompt_lengths, viz_confusion_matrix
from scripts.memory_utils import (
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

# ---------- Visualization helpers ----------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def viz_prompt_lengths(run_dir: Path, avg_stats: Dict[str, Dict[str, float]]) -> None:
    if go is None or plotly_offline_plot is None:
        return
    metrics = ["chars", "words", "lines"]
    wm = [avg_stats["with_memory"][m] for m in metrics]
    bl = [avg_stats["baseline"][m] for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="with_memory", x=metrics, y=wm))
    fig.add_trace(go.Bar(name="baseline", x=metrics, y=bl))
    fig.update_layout(barmode="group", title="Average prompt sizes", yaxis_title="Count")
    plotly_offline_plot(fig, filename=str(run_dir / "prompt_lengths.html"), auto_open=False, include_plotlyjs="cdn")


def viz_confusion_matrix(
    run_dir: Path,
    title: str,
    roles: List[str],
    confusion: Dict[str, Dict[str, float]],
    f1_by_role: Dict[str, float] | None,
    filename: str,
) -> None:
    if go is None or plotly_offline_plot is None:
        return
    n = len(roles)
    # Build count matrix
    M = np.zeros((n, n), dtype=float)
    for i, tr in enumerate(roles):
        row = confusion.get(tr, {}) or {}
        for j, pr in enumerate(roles):
            M[i, j] = float(row.get(pr, 0.0))
    # Normalize rows to probabilities (avoid division by zero)
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, np.where(row_sums == 0, 1.0, row_sums), where=np.ones_like(M, dtype=bool))
    # Text annotations: show percentage and (count). On diagonal, also show F1 if provided
    text = []
    for i in range(n):
        row = []
        for j in range(n):
            pct = norm[i, j] * 100.0
            cell = f"{pct:.1f}%\n({int(M[i, j])})"
            if i == j and f1_by_role is not None:
                role = roles[i]
                f1 = float(f1_by_role.get(role, 0.0)) if role in f1_by_role else 0.0
                cell = f"{pct:.1f}%\n({int(M[i, j])})\nF1={f1:.2f}"
            row.append(cell)
        text.append(row)
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
        colorbar=dict(title="Row-normalized")
    ))
    fig.update_layout(title=title, xaxis_title="Predicted role", yaxis_title="True role")
    plotly_offline_plot(fig, filename=str(run_dir / filename), auto_open=False, include_plotlyjs="cdn")



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

    # For belief baseline: maintain per-game belief vectors as dict[player] -> role
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

        # Retrieval-only metrics removed in LLM-only mode

        # Prompts and stats
        retrieved_pairs = [(all_entries[j], float(s)) for j, s in zip(topk_idx, topk_scores)]
        # Classification task suffix (instructions + JSON schema requirement)
        roles_list = ", ".join(valid_roles) if valid_roles else ""
        task_suffix = "\n\nTask: Predict the role of the player who proposed the party."
        if roles_list:
            task_suffix += f" Valid roles: {roles_list}."
        task_suffix += (
            " Respond ONLY with a single-line JSON object matching this schema: "
            "{\"role\": \"<ROLE>\"}. Do not include explanations, prefixes, or extra keys."
        )

        # Build prompts: memory-augmented is unchanged; baseline depends on baseline_mode
        prompt_mem_task, mem_meta = assemble_prompt_with_meta(target, retrieved_pairs, task_suffix=task_suffix)

        gid = target.game_id
        # Default baseline: full transcript template (legacy behavior)
        if baseline_mode == "full_transcript":
            prompt_base_task, base_meta = assemble_baseline_prompt_with_meta(target, task_suffix=task_suffix)
        else:
            # belief_vector baseline: build compact state + belief prompt
            quests = game_quests.get(gid, {})
            players = game_players.get(gid, [])
            # Ensure stable belief state dict for this game
            beliefs = belief_state_by_game[gid]
            for p in players:
                beliefs.setdefault(p, "unknown")
            # Past state lines for quests < current quest
            past_state_lines: List[str] = []
            for q in sorted(quests.keys()):
                if int(q) >= int(target.quest):
                    continue
                past_state_lines.append(quest_state_line(gid, int(q), quests[q]))
            current_msgs = quests.get(int(target.quest), [])
            current_transcript = quest_transcript_only(current_msgs)
            prompt_base_task = assemble_belief_baseline_prompt(
                game_id=gid,
                players=players,
                belief_vector=beliefs,
                past_state_lines=past_state_lines,
                current_quest=int(target.quest),
                current_transcript=current_transcript,
                valid_roles=valid_roles,
            )
            base_meta = {"mode": "belief_vector"}

        pstats = prompt_stats(prompt_mem_task, prompt_base_task)

        # Optional LLM call (uses augmented or baseline prompt depending on run mode)
        llm_prompt = prompt_base_task if bool(use_llm) and bool(llm_use_baseline_prompt) else prompt_mem_task
        llm_out = llm_role_predict(llm_prompt, bool(use_llm), openai_model, openai_api_key)

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
                    pchars=len(llm_prompt),
                    error=repr(err),
                    warning=repr(warn),
                    preview=repr(preview),
                )
            )

        # Proposer-role prediction (LLM or heuristic fallback)
        true_role = target.proposer_role
        pred_role = None
        # Debug info for processed output
        processed_used_repair = False
        processed_beliefs_preview: str | None = None
        proposer_from_beliefs: str | None = None
        if bool(use_llm):
            raw_pred = llm_out.get("prediction")
            pred_role = extract_role_label(raw_pred, valid_roles)
            # If using belief baseline, parse/update beliefs robustly and derive proposer role
            if baseline_mode == "belief_vector" and llm_use_baseline_prompt:
                beliefs_obj: Dict[str, Any] | None = None
                # 1) Try robust local extraction
                beliefs_container = extract_beliefs_object(raw_pred or "")
                if isinstance(beliefs_container, dict):
                    beliefs_obj = beliefs_container.get("beliefs") if isinstance(beliefs_container.get("beliefs"), dict) else None
                # 2) If still missing, attempt a one-shot TypeChat-style repair
                if beliefs_obj is None:
                    schema_hint = '{"beliefs": {"player-1": "<ROLE>", "player-2": "<ROLE>", ...}}'
                    repaired = typechat_repair_to_json(raw_pred or "", schema_hint, openai_model, openai_api_key, True)
                    repaired_container = extract_beliefs_object(repaired or "") if repaired else None
                    if isinstance(repaired_container, dict):
                        beliefs_obj = repaired_container.get("beliefs") if isinstance(repaired_container.get("beliefs"), dict) else None
                        processed_used_repair = True
                # 3) Apply updates and set pred_role from proposer
                if isinstance(beliefs_obj, dict):
                    beliefs = belief_state_by_game[gid]
                    for p, r in beliefs_obj.items():
                        if isinstance(r, str):
                            beliefs[str(p)] = r.strip()
                    # Build a compact preview of the parsed beliefs JSON for debugging
                    try:
                        processed_beliefs_preview = json.dumps({"beliefs": beliefs_obj})[:160]
                    except Exception:
                        processed_beliefs_preview = None
                    if isinstance(target.proposer, str):
                        pr = beliefs.get(target.proposer)
                        if isinstance(pr, str) and any(pr.lower() == vr.lower() for vr in valid_roles):
                            pred_role = next(vr for vr in valid_roles if pr.lower() == vr.lower())
                            proposer_from_beliefs = pred_role
                # If still no pred_role (repair may have produced role-only JSON), try role repair
                if pred_role is None:
                    schema_hint_role = '{"role": "<ROLE>"}'
                    repaired_role = typechat_repair_to_json(raw_pred or "", schema_hint_role, openai_model, openai_api_key, True)
                    pr2 = extract_role_label(repaired_role, valid_roles) if repaired_role else None
                    if pr2 is not None and pred_role is None:
                        processed_used_repair = True
                    pred_role = pred_role or pr2
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
    ap.add_argument("--k", type=int, default=3, help="Top-k for retrieval and Recall@k")
    ap.add_argument("--memory_format", type=str, default="template", choices=["template", "template+summary"], help="Memory format for entries")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer embedding model name")
    ap.add_argument("--outdir", type=str, default="outputs/eval", help="Base output directory for saving results and visuals")
    ap.add_argument("--llm", action="store_true", help="Call an LLM for downstream role prediction (TypeChat-like JSON output enforced)")
    ap.add_argument("--llm_model", type=str, default="ollama:llama2:13b", help="LLM identifier. Use 'ollama:<model>' (e.g., 'ollama:llama3:8b-instruct') or 'openai:<model>' for OpenAI.")
    ap.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key (if --llm)")
    ap.add_argument("--max_games", type=int, default=None, help="Limit number of games for quick tests")
    ap.add_argument("--num_runs", type=int, default=1, help="Repeat the experiment N times and average results")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed (incremented per run)")
    ap.add_argument("--baseline_mode", type=str, default="full_transcript", choices=["full_transcript", "belief_vector"], help="Baseline prompting mode: full transcript per round or iterative belief-vector baseline.")
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
    run_dir = Path(args.outdir) / f"{ts}_k-{args.k}_mem-{args.memory_format}_llm"
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
        print("  Evaluating baseline (no memory in prompt), memory_format=template")
        agg_base_i, res_base_i = evaluate_config(
            files=files,
            k=int(args.k),
            memory_format="template",
            model_name=args.model,
            use_llm=bool(args.llm),
            openai_model=args.llm_model,
            openai_api_key=args.openai_api_key,
            llm_use_baseline_prompt=True,
            baseline_mode=str(args.baseline_mode),
        )
        agg_base_runs.append(agg_base_i)
        with (run_subdir / "results_baseline.json").open("w", encoding="utf-8") as f:
            json.dump(res_base_i, f, ensure_ascii=False, indent=2)
        with (run_subdir / "aggregate_baseline.json").open("w", encoding="utf-8") as f:
            json.dump(agg_base_i, f, ensure_ascii=False, indent=2)

        # Current
        print(f"  Evaluating current (with memory in prompt), memory_format={args.memory_format}")
        agg_cur_i, res_cur_i = evaluate_config(
            files=files,
            k=int(args.k),
            memory_format=args.memory_format,
            model_name=args.model,
            use_llm=bool(args.llm),
            openai_model=args.llm_model,
            openai_api_key=args.openai_api_key,
            llm_use_baseline_prompt=False,
            baseline_mode=str(args.baseline_mode),
        )
        agg_cur_runs.append(agg_cur_i)
        with (run_subdir / "results_current.json").open("w", encoding="utf-8") as f:
            json.dump(res_cur_i, f, ensure_ascii=False, indent=2)
        with (run_subdir / "aggregate_current.json").open("w", encoding="utf-8") as f:
            json.dump(agg_cur_i, f, ensure_ascii=False, indent=2)

    # Averages across runs
    agg_base = average_aggregates(agg_base_runs)
    agg_cur = average_aggregates(agg_cur_runs)
    with (run_dir / "aggregate_baseline_mean.json").open("w", encoding="utf-8") as f:
        json.dump(agg_base, f, ensure_ascii=False, indent=2)
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
