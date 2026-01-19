from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


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


@dataclass
class PlayerVector:
    """Compact, interpretable world-model state per player."""

    # Main belief: probability player is evil
    p_evil: float = 0.5

    # Simple behavioral signals (EMA in [0,1])
    talkativeness: float = 0.0


VectorState = Dict[str, PlayerVector]


_party_re = re.compile(r"proposed a party:\s*(.*)$")


def init_vector_state(players: List[str]) -> VectorState:
    return {p: PlayerVector() for p in players}


def ensure_players(state: VectorState, players: List[str]) -> None:
    for p in players:
        if p not in state:
            state[p] = PlayerVector()


def update_from_transcript(
    state: VectorState,
    players: List[str],
    transcript_messages: List[Dict[str, Any]],
    alpha: float = 0.3,
) -> None:
    """Update lightweight behavioral signals from a quest transcript.

    Currently only talkativeness (normalized message share) is implemented.
    """

    ensure_players(state, players)

    counts: Dict[str, int] = {p: 0 for p in players}
    total = 0
    for m in transcript_messages:
        speaker = m.get("speaker") or m.get("player") or m.get("name")
        if not speaker:
            continue
        speaker = str(speaker)
        if speaker.lower() == "system":
            continue
        if speaker in counts:
            counts[speaker] += 1
            total += 1

    if total <= 0:
        return

    for p in players:
        share = counts.get(p, 0) / float(total)
        pv = state[p]
        pv.talkativeness = (1.0 - alpha) * pv.talkativeness + alpha * float(share)
        pv.talkativeness = _clamp(pv.talkativeness, 0.0, 1.0)


def update_from_quest_outcome(
    state: VectorState,
    players: List[str],
    quest_team: List[str],
    quest_failed: Optional[bool],
    w_fail: float = 0.8,
    w_success: float = -0.4,
) -> None:
    """Bayesian-ish log-odds update of p_evil based on quest outcome.

    - If quest_failed is True: increase suspicion of team members.
    - If quest_failed is False: decrease suspicion of team members.
    - If quest_failed is None: no-op.

    This is intentionally simple and stable for v0.
    """

    ensure_players(state, players)

    if quest_failed is None:
        return

    delta = float(w_fail if quest_failed else w_success)
    for p in quest_team:
        if p not in state:
            continue
        pv = state[p]
        log_odds = _logit(pv.p_evil) + delta
        pv.p_evil = _clamp(_sigmoid(log_odds), 0.01, 0.99)


def render_vector_memory(state: VectorState, players: List[str], max_players: int = 12) -> str:
    """Render the vector memory as a compact, prompt-friendly block."""

    # stable order: given players list first, then any extras
    ordered = list(players)
    for p in state.keys():
        if p not in ordered:
            ordered.append(p)

    lines = ["VECTOR_MEMORY (per player)"]
    for p in ordered[:max_players]:
        pv = state[p]
        lines.append(
            f"- {p}: p_evil={pv.p_evil:.2f}, talk={pv.talkativeness:.2f}"
        )
    return "\n".join(lines)


def default_max_players() -> int:
    try:
        return int(os.getenv("VECTOR_MEMORY_MAX_PLAYERS", "12"))
    except Exception:
        return 12


def parse_party_and_outcome_from_msgs(msgs: List[Dict[str, Any]]) -> tuple[list[str], Optional[bool]]:
    """Best-effort extraction of (party_members, quest_failed) from Avalon message dicts."""

    party_members: list[str] = []
    quest_failed: Optional[bool] = None
    for m in msgs:
        player = str(m.get("player", ""))
        msg = str(m.get("msg", ""))
        if player != "system":
            continue
        if "proposed a party:" in msg:
            mm = _party_re.search(msg)
            if mm:
                party_members = [s.strip() for s in mm.group(1).split(",") if s.strip()]
        if msg.endswith("quest failed!"):
            quest_failed = True
        elif msg.endswith("quest succeeded!"):
            quest_failed = False
    return party_members, quest_failed


def summarize_state(state: VectorState, players: List[str], topk: int = 3) -> Dict[str, Any]:
    """Small structured summary for logging/analysis."""

    ordered = [p for p in players if p in state]
    suspects = sorted(ordered, key=lambda p: state[p].p_evil, reverse=True)
    return {
        "top_suspects": [
            {"player": p, "p_evil": float(state[p].p_evil), "talk": float(state[p].talkativeness)}
            for p in suspects[: max(0, int(topk))]
        ],
        "min_p_evil": float(min((state[p].p_evil for p in ordered), default=0.0)),
        "max_p_evil": float(max((state[p].p_evil for p in ordered), default=0.0)),
    }


def state_preview_json(state: VectorState, players: List[str], limit_chars: int = 240) -> str:
    payload = {p: asdict(state[p]) for p in players if p in state}
    s = json.dumps(payload, ensure_ascii=False)
    return s[:limit_chars]
