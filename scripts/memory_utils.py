#!/usr/bin/env python3
from __future__ import annotations
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

@dataclass
class MemoryEntry:
    game_id: str
    quest: int
    text: str
    entry_id: str
    proposer: str | None = None
    proposer_role: str | None = None


# ---------- Data loading and memory formats ----------

def load_game_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def messages_by_quest(game: Dict) -> Dict[int, List[Dict]]:
    msgs = list(game.get("messages", {}).values())
    msgs.sort(key=lambda m: (int(m.get("quest", 0)), int(m.get("turn", 0)), str(m.get("mid", ""))))
    grouped: Dict[int, List[Dict]] = {}
    for m in msgs:
        q = int(m.get("quest", 0))
        grouped.setdefault(q, []).append(m)
    return grouped


def quest_state_line(game_id: str, quest: int, msgs: List[Dict]) -> str:
    """Build a compact state line for a given quest.

    Follows the paper-style template: `quest-i: oi (party: pj | player votes: vj)`.
    """
    party_members: List[str] = []
    votes: List[str] = []
    outcome: str = "unknown"
    for m in msgs:
        player = m.get("player", "")
        msg = m.get("msg", "")
        if player == "system":
            if "proposed a party:" in msg:
                p = extract_party(msg)
                if p:
                    party_members = [s.strip() for s in p.split(",") if s.strip()]
            if msg.startswith("party vote outcome:"):
                # Raw system line already describes votes; keep compact suffix
                votes.append(msg.replace("party vote outcome:", "").strip())
            if msg.endswith("quest succeeded!"):
                outcome = "success"
            if msg.endswith("quest failed!"):
                outcome = "fail"
    party_str = ", ".join(party_members) if party_members else "unknown"
    votes_str = "; ".join(votes) if votes else "unknown"
    return f"quest-{quest}: {outcome} (party: {party_str} | player votes: {votes_str})"


def quest_transcript_only(msgs: List[Dict]) -> str:
    """Return only the player chat transcript for a quest (no system lines)."""
    lines: List[str] = []
    for m in msgs:
        player = m.get("player", "")
        if player == "system":
            continue
        msg = m.get("msg", "")
        turn = m.get("turn", "?")
        lines.append(f"[Turn {turn}] {player}: {msg}")
    return "\n".join(lines)

_party_re = re.compile(r"proposed a party:\s*(.*)$")

def extract_party(line: str) -> str | None:
    m = _party_re.search(line)
    if m:
        return m.group(1).strip()
    return None


def build_template_entry_for_quest(game: Dict, game_id: str, quest: int, msgs: List[Dict]) -> MemoryEntry:
    system_lines: List[str] = []
    transcript_lines: List[str] = []
    proposed_party: str | None = None
    vote_outcome: str | None = None
    vote_result: str | None = None
    quest_result: str | None = None
    proposer: str | None = None
    proposer_role: str | None = None

    # Build helper map from player name to role (if available)
    name_to_role: Dict[str, str] = {}
    try:
        for u in (game.get("users", {}) or {}).values():
            name = u.get("name")
            role = u.get("role")
            if name and role:
                name_to_role[str(name)] = str(role)
    except Exception:
        pass

    for m in msgs:
        player = m.get("player", "")
        msg = m.get("msg", "")
        turn = m.get("turn", "?")
        if player == "system":
            system_lines.append(msg)
            if "proposed a party:" in msg:
                proposed_party = extract_party(msg)
                # Try to parse proposer name before " proposed a party:"
                try:
                    # Expected format: "player-4 proposed a party: ..."
                    proposer_token = msg.split(" proposed a party:", 1)[0].strip()
                    proposer = proposer_token or proposer
                    if proposer and proposer_role is None:
                        proposer_role = name_to_role.get(proposer)
                except Exception:
                    pass
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
            transcript_lines.append(f"[Turn {turn}] {player}: {msg}")

    header = [
        f"GAME {game_id} | QUEST {quest}",
        "TEMPLATE SUMMARY:",
        f"- Proposed party: {proposed_party or 'unknown'}",
        f"- Party vote: {vote_outcome or 'unknown'}",
        f"- Vote result: {vote_result or 'unknown'}",
        f"- Quest result: {quest_result or 'unknown'}",
        "",
        "Transcript (player messages only):",
    ]
    text = "\n".join(header + transcript_lines)
    return MemoryEntry(
        game_id=game_id,
        quest=int(quest),
        text=text,
        entry_id=f"{game_id}_q{quest}",
        proposer=proposer,
        proposer_role=proposer_role,
    )


def heuristic_summary(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    party = "unknown"
    vote = "unknown"
    result = "unknown"
    n_msgs = 0
    speakers: Dict[str, int] = defaultdict(int)
    for ln in lines:
        if ln.startswith("- Proposed party:"):
            party = ln.split(":", 1)[-1].strip()
        elif ln.startswith("- Party vote:"):
            vote = ln.split(":", 1)[-1].strip()
        elif ln.startswith("- Quest result:"):
            result = ln.split(":", 1)[-1].strip()
        elif ln.startswith("[Turn "):
            n_msgs += 1
            try:
                rest = ln.split("]", 1)[-1].strip()
                sp = rest.split(":", 1)[0]
                speakers[sp] += 1
            except Exception:
                pass
    top_speakers = ", ".join([f"{s}({c})" for s, c in sorted(speakers.items(), key=lambda x: -x[1])[:3]]) or "none"
    return ("\n".join([
        "HEURISTIC SUMMARY:",
        f"- Party: {party}",
        f"- Vote: {vote}",
        f"- Result: {result}",
        f"- Messages: {n_msgs}",
        f"- Top speakers: {top_speakers}",
    ]) + "\n")


def build_memory_entries(game: Dict, game_id: str, memory_format: str = "template") -> List[MemoryEntry]:
    grouped = messages_by_quest(game)
    entries: List[MemoryEntry] = []
    for quest, msgs in grouped.items():
        entry = build_template_entry_for_quest(game, game_id, quest, msgs)
        if memory_format == "template+summary":
            entry.text = entry.text + "\n" + heuristic_summary(entry.text)
        entries.append(entry)
    return entries


# ---------- Embeddings and indexing ----------

def build_embeddings(entries: List[MemoryEntry], model_name: str) -> Tuple[np.ndarray, Any]:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. pip install -r requirements.txt")
    model = SentenceTransformer(model_name)
    corpus = [e.text for e in entries]
    emb = model.encode(corpus, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32"), model


def build_faiss_ip_index(embeddings: np.ndarray) -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu is required. pip install -r requirements.txt")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def retrieve_top(index: Any, query_vec: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    D, I = index.search(query_vec.astype("float32"), topn)
    return D[0], I[0]


# ---------- Retrieval policies ----------

def temporal_weight(score: float, target: MemoryEntry, candidate: MemoryEntry, alpha: float = 0.5) -> float:
    dist = abs(int(target.quest) - int(candidate.quest))
    if target.game_id != candidate.game_id:
        dist += 2
    return float(score) * math.exp(-alpha * float(dist))


def rerank_temporal(entries: List[MemoryEntry], indices: Iterable[int], scores: Iterable[float], target_entry: MemoryEntry, alpha: float) -> List[Tuple[int, float]]:
    pairs = []
    for j, s in zip(indices, scores):
        adj = temporal_weight(s, target_entry, entries[int(j)], alpha)
        pairs.append((int(j), float(adj)))
    pairs.sort(key=lambda x: -x[1])
    return pairs




# ---------- Math helper ----------

def pca_2d(x: np.ndarray, center: bool = True) -> np.ndarray:
    X = x.astype(np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:2].T
    return X @ comps
