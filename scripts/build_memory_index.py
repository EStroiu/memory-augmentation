#!/usr/bin/env python3
"""
Tiny utility to:
- Load one Avalon game JSON (or default sample)
- Produce a template-only memory entry per quest (round)
- Build a FAISS Flat index of sentence-transformer embeddings
- Retrieve top-k similar chunks for a chosen round
- Print the prompt that would be sent to an LLM (call is stubbed for dev)

Usage (from repository root):
    python scripts/build_memory_index.py \
        --data streamlit/sample.json \
        --round 2 \
        --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except ImportError as e:
    print("This script requires numpy. Please install requirements: pip install -r requirements.txt", file=sys.stderr)
    raise

# Lazy import ML deps to give a clearer error if missing
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

# Plotly is optional for visualizations; warn if missing
try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.offline import plot as plotly_offline_plot  # type: ignore
except Exception:
    go = None  # type: ignore
    plotly_offline_plot = None  # type: ignore


@dataclass
class MemoryEntry:
    game_id: str
    quest: int
    text: str
    entry_id: str


def load_game_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def messages_by_quest(game: Dict) -> Dict[int, List[Dict]]:
    """Group message dicts by quest, preserving increasing turn order.
    Expects game["messages"] to be a dict keyed by string ids.
    """
    msgs = list(game.get("messages", {}).values())
    # sort by quest then turn then fallback by mid string for stability
    msgs.sort(key=lambda m: (int(m.get("quest", 0)), int(m.get("turn", 0)), str(m.get("mid", ""))))
    grouped: Dict[int, List[Dict]] = {}
    for m in msgs:
        q = int(m.get("quest", 0))
        grouped.setdefault(q, []).append(m)
    return grouped


_party_re = re.compile(r"proposed a party:\s*(.*)$")


def extract_party(line: str) -> str | None:
    m = _party_re.search(line)
    if m:
        return m.group(1).strip()
    return None


def build_memory_entries(game: Dict, game_id: str) -> List[MemoryEntry]:
    grouped = messages_by_quest(game)
    entries: List[MemoryEntry] = []
    for quest, msgs in grouped.items():
        system_lines: List[str] = []
        transcript_lines: List[str] = []
        proposed_party: str | None = None
        vote_outcome: str | None = None
        vote_result: str | None = None
        quest_result: str | None = None

        for m in msgs:
            player = m.get("player", "")
            msg = m.get("msg", "")
            turn = m.get("turn", "?")
            if player == "system":
                system_lines.append(msg)
                if "proposed a party:" in msg:
                    proposed_party = extract_party(msg)
                if msg.startswith("party vote outcome:"):
                    vote_outcome = msg
                if msg.endswith("vote succeeded!") or msg.endswith("vote succeeded! initiating quest vote!"):
                    vote_result = "vote succeeded"
                if msg.endswith("vote failed!"):
                    vote_result = "vote failed"
                if msg.endswith("quest succeeded!"):
                    quest_result = "quest succeeded"
                if msg.endswith("quest failed!"):
                    quest_result = "quest failed"
            else:
                # Only include player messages in transcript template
                transcript_lines.append(f"[Turn {turn}] {player}: {msg}")

        header = [
            f"GAME {game_id} | QUEST {quest}",
            "TEMPLATE SUMMARY (no LLM, structured only):",
            f"- Proposed party: {proposed_party or 'unknown'}",
            f"- Party vote: {vote_outcome or 'unknown'}",
            f"- Vote result: {vote_result or 'unknown'}",
            f"- Quest result: {quest_result or 'unknown'}",
            "",
            "Transcript (player messages only):",
        ]
        text = "\n".join(header + transcript_lines)
        entries.append(MemoryEntry(game_id=game_id, quest=int(quest), text=text, entry_id=f"{game_id}_q{quest}"))

    return entries


def build_embeddings(entries: List[MemoryEntry], model_name: str) -> Tuple[np.ndarray, Any]:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Install via: pip install -r requirements.txt")
    model = SentenceTransformer(model_name)
    corpus = [e.text for e in entries]
    emb = model.encode(corpus, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32"), model


def build_faiss_ip_index(embeddings: np.ndarray) -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu is required. Install via: pip install -r requirements.txt")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def retrieve_top_k(index: Any, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    D, I = index.search(query_vec.astype("float32"), k)
    return D[0], I[0]


def assemble_prompt(target: MemoryEntry, retrieved: List[Tuple[MemoryEntry, float]]) -> str:
    parts = []
    parts.append("System: You are helping analyze an Avalon game round. Use the memory context to inform your reasoning.")
    parts.append("")
    parts.append(f"Target: GAME {target.game_id} | QUEST {target.quest}")
    parts.append("")
    parts.append("Memory context (top-k similar rounds):")
    for i, (entry, score) in enumerate(retrieved, start=1):
        parts.append(f"---- Memory {i} (cosine sim ~ {score:.3f}) | {entry.entry_id}")
        parts.append(entry.text)
        parts.append("")
    parts.append("Your task: Provide an analysis or next-step reasoning for the target round using the context above.")
    parts.append("Answer:")
    return "\n".join(parts)


def assemble_baseline_prompt(target: MemoryEntry) -> str:
    """Baseline prompt that includes ONLY the target round's own template/transcript, no external memory."""
    parts = []
    parts.append("System: You are helping analyze an Avalon game round. No external memory is provided for this baseline.")
    parts.append("")
    parts.append(f"Target: GAME {target.game_id} | QUEST {target.quest}")
    parts.append("")
    parts.append("Target round context:")
    parts.append(target.text)
    parts.append("")
    parts.append("Your task: Provide an analysis or next-step reasoning for the target round.")
    parts.append("Answer:")
    return "\n".join(parts)


def pick_default_data_path(repo_root: Path) -> Path:
    # Prefer the small sample if present
    p = repo_root / "streamlit" / "sample.json"
    if p.exists():
        return p
    # Fallback to any file in dataset/
    ds = repo_root / "dataset"
    if ds.exists():
        for cand in sorted(ds.glob("*.json")):
            return cand
    raise FileNotFoundError("Could not locate a sample data file. Expected streamlit/sample.json or dataset/*.json")


def ensure_run_dir(base_outdir: Path, game_id: str, round_id: int, k: int, model_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name.split("/")[-1])
    run_dir = base_outdir / f"{ts}_game-{game_id}_round-{round_id}_k-{k}_model-{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def try_save_faiss_index(index: Any, path: Path) -> bool:
    if faiss is None:
        return False
    try:
        faiss.write_index(index, str(path))
        return True
    except Exception:
        return False


def pca_2d(x: np.ndarray, center: bool = True) -> np.ndarray:
    """Simple PCA to 2D using numpy only."""
    X = x.astype(np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # covariance via SVD directly for numerical stability
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:2].T  # (d,2)
    return X @ comps  # (n,2)


def viz_topk_bar(run_dir: Path, retrieved_pairs: List[Tuple[MemoryEntry, float]]) -> None:
    if go is None or plotly_offline_plot is None:
        return
    labels = [f"q{e.quest}" for e, _ in retrieved_pairs]
    scores = [float(s) for _, s in retrieved_pairs]
    fig = go.Figure(data=[go.Bar(x=labels, y=scores)])
    fig.update_layout(title="Top-k cosine similarities", xaxis_title="Quest", yaxis_title="Cosine similarity (target vs entry)")
    plotly_offline_plot(fig, filename=str(run_dir / "topk_bar.html"), auto_open=False, include_plotlyjs="cdn")


def viz_embedding_2d(run_dir: Path, emb: np.ndarray, entries: List[MemoryEntry], target_idx: int, retrieved_indices: List[int]) -> None:
    if go is None or plotly_offline_plot is None:
        return
    pts = pca_2d(emb)
    x, y = pts[:, 0], pts[:, 1]
    colors = []
    sizes = []
    texts = []
    retrieved_set = set(retrieved_indices)
    for i, e in enumerate(entries):
        if i == target_idx:
            colors.append("red")
            sizes.append(14)
        elif i in retrieved_set:
            colors.append("orange")
            sizes.append(10)
        else:
            colors.append("steelblue")
            sizes.append(8)
        texts.append(f"{e.entry_id} (q{e.quest})")
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="markers", marker=dict(color=colors, size=sizes), text=texts, hoverinfo="text")])
    fig.update_layout(title="Round embeddings (PCA 2D): red=target, orange=retrieved")
    plotly_offline_plot(fig, filename=str(run_dir / "embedding_2d.html"), auto_open=False, include_plotlyjs="cdn")


def viz_similarity_hist(run_dir: Path, sims: np.ndarray) -> None:
    if go is None or plotly_offline_plot is None:
        return
    fig = go.Figure(data=[go.Histogram(x=sims, nbinsx=20)])
    fig.update_layout(title="Similarity distribution (target vs all rounds)", xaxis_title="Cosine similarity", yaxis_title="Count")
    plotly_offline_plot(fig, filename=str(run_dir / "similarity_hist.html"), auto_open=False, include_plotlyjs="cdn")


def compute_prompt_stats(prompt_with_memory: str, prompt_baseline: str) -> Dict[str, Dict[str, int]]:
        def _stats(t: str) -> Dict[str, int]:
                return {
                        "chars": len(t),
                        "words": len(t.split()),
                        "lines": t.count("\n") + (1 if t else 0),
                }
        return {
                "with_memory": _stats(prompt_with_memory),
                "baseline": _stats(prompt_baseline),
        }


def viz_prompt_lengths(run_dir: Path, stats: Dict[str, Dict[str, int]]) -> None:
        if go is None or plotly_offline_plot is None:
                return
        metrics = ["chars", "words", "lines"]
        wm = [stats["with_memory"][m] for m in metrics]
        bl = [stats["baseline"][m] for m in metrics]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="with_memory", x=metrics, y=wm))
        fig.add_trace(go.Bar(name="baseline", x=metrics, y=bl))
        fig.update_layout(barmode="group", title="Prompt size comparison", yaxis_title="Count")
        plotly_offline_plot(fig, filename=str(run_dir / "prompt_lengths.html"), auto_open=False, include_plotlyjs="cdn")


def save_prompt_comparison_html(run_dir: Path, prompt_with_memory: str, prompt_baseline: str) -> None:
        html = f"""
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>Prompt Comparison</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }}
        .panel {{ border: 1px solid #ccc; border-radius: 6px; overflow: hidden; }}
        .panel h2 {{ margin: 0; padding: 12px; background: #f5f5f5; font-size: 16px; }}
        .panel pre {{ margin: 0; padding: 12px; white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.4; }}
    </style>
    </head>
<body>
    <div class=\"grid\">
        <div class=\"panel\">
            <h2>Baseline (no memory)</h2>
            <pre>{prompt_baseline}</pre>
        </div>
        <div class=\"panel\">
            <h2>With Memory (top-k)</h2>
            <pre>{prompt_with_memory}</pre>
        </div>
    </div>
</body>
</html>
"""
        (run_dir / "prompt_compare.html").write_text(html, encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query a FAISS memory index over Avalon rounds. Saves prompts and visuals.")
    parser.add_argument("--data", type=str, default=None, help="Path to a game JSON file (default: streamlit/sample.json or first in dataset/)")
    parser.add_argument("--round", dest="round_id", type=int, default=1, help="Quest/round id to query (e.g., 1..5)")
    parser.add_argument("--k", type=int, default=3, help="Top-k similar rounds to retrieve")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    parser.add_argument("--outdir", type=str, default="outputs/runs", help="Base output directory for saving results and visuals")
    parser.add_argument("--exclude-self", action="store_true", help="Exclude the target round from retrieval results")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data) if args.data else pick_default_data_path(repo_root)

    game = load_game_json(data_path)
    game_id = data_path.stem

    print(f"Loaded game: {game_id} from {data_path}")

    entries = build_memory_entries(game, game_id)
    if not entries:
        print("No entries found in the game file.", file=sys.stderr)
        return 1

    # Build embeddings and index
    print(f"Encoding {len(entries)} round entries with model: {args.model}")
    emb, model = build_embeddings(entries, args.model)
    index = build_faiss_ip_index(emb)

    # Find the target round entry
    try:
        target_idx = next(i for i, e in enumerate(entries) if int(e.quest) == int(args.round_id))
    except StopIteration:
        print(f"Round/quest {args.round_id} not found. Available: {[e.quest for e in entries]}", file=sys.stderr)
        return 2

    target_entry = entries[target_idx]

    # Embed target (normalize to cosine/IP)
    target_vec = model.encode([target_entry.text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")

    # Retrieve
    D, I = retrieve_top_k(index, target_vec, k=min(args.k + (1 if args.exclude_self else 0), len(entries)))
    # Build retrieved pairs, optionally filtering out self
    pairs: List[Tuple[MemoryEntry, float, int]] = []  # (entry, score, idx)
    for rank, (j, score) in enumerate(zip(I, D)):
        idx = int(j)
        if args.exclude_self and idx == target_idx:
            continue
        pairs.append((entries[idx], float(score), idx))
        if len(pairs) == args.k:
            break
    retrieved_pairs: List[Tuple[MemoryEntry, float]] = [(e, s) for (e, s, _) in pairs]
    retrieved_indices: List[int] = [idx for (_, _, idx) in pairs]

    # Assemble and print prompt
    prompt = assemble_prompt(target_entry, retrieved_pairs)
    baseline_prompt = assemble_baseline_prompt(target_entry)

    # Print a short report first
    print("")
    print("Top-k retrieved entries (id, quest, score):")
    for entry, score in retrieved_pairs:
        print(f"  - {entry.entry_id} | quest={entry.quest} | score={score:.3f}")

    print("")
    print("Prompt to LLM (stub):")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print("[Stub] LLM call would be executed here.")

    # Save artifacts to timestamped run dir
    run_dir = ensure_run_dir(Path(args.outdir), game_id, int(args.round_id), int(args.k), args.model)
    print(f"\nSaving artifacts to: {run_dir}")

    # Config and metadata
    config = {
        "data_path": str(data_path),
        "game_id": game_id,
        "round": int(args.round_id),
        "k": int(args.k),
        "model": args.model,
        "exclude_self": bool(args.exclude_self),
    }
    save_json(run_dir / "config.json", config)

    # Entries and embeddings
    entries_serialized = [asdict(e) for e in entries]
    save_json(run_dir / "entries.json", entries_serialized)
    np.save(run_dir / "embeddings.npy", emb)

    # Index
    if try_save_faiss_index(index, run_dir / "index.faiss"):
        pass  # saved successfully

    # Retrieval
    retrieval = {
        "target_index": int(target_idx),
        "target_entry_id": target_entry.entry_id,
        "results": [
            {"entry_id": e.entry_id, "quest": int(e.quest), "score": float(s)} for (e, s) in retrieved_pairs
        ],
    }
    save_json(run_dir / "retrieval.json", retrieval)

    # Prompts
    save_text(run_dir / "prompt_with_memory.txt", prompt)
    save_text(run_dir / "prompt_baseline.txt", baseline_prompt)
    # Prompt stats + comparison visuals
    prompt_stats = compute_prompt_stats(prompt, baseline_prompt)
    save_json(run_dir / "prompt_stats.json", prompt_stats)
    viz_prompt_lengths(run_dir, prompt_stats)
    save_prompt_comparison_html(run_dir, prompt, baseline_prompt)
    # Dashboard removed by request; no index.html generated

    # Visualizations
    # 1) top-k bar
    viz_topk_bar(run_dir, retrieved_pairs)
    # 2) embedding 2D
    viz_embedding_2d(run_dir, emb, entries, target_idx, retrieved_indices)
    # 3) sim distribution vs all
    sims_all = (emb @ target_vec.astype("float32")).astype("float32")  # cosine due to normalization
    viz_similarity_hist(run_dir, sims_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

