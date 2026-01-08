from __future__ import annotations

import json
import math
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


def state_preview_json(state: VectorState, players: List[str], limit_chars: int = 240) -> str:
    payload = {p: asdict(state[p]) for p in players if p in state}
    s = json.dumps(payload, ensure_ascii=False)
    return s[:limit_chars]
