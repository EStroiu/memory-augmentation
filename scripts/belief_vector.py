from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from scripts.json_utils import extract_role_label


BeliefStateByGame = Dict[str, Dict[str, str]]
BeliefConfidenceByGame = Dict[str, Dict[str, float]]


def ensure_belief_state_for_game(
    belief_state_by_game: BeliefStateByGame,
    game_id: str,
    players: List[str],
    default_role: str = "unknown",
) -> Dict[str, str]:
    """Get/create belief state for one game and ensure all players exist."""
    beliefs = belief_state_by_game.setdefault(str(game_id), {})
    for player in players:
        beliefs.setdefault(str(player), default_role)
    return beliefs


def ensure_belief_confidence_for_game(
    belief_confidence_by_game: BeliefConfidenceByGame,
    game_id: str,
    players: List[str],
    default_confidence: float = 0.0,
) -> Dict[str, float]:
    """Get/create per-player confidence map for one game."""
    conf = belief_confidence_by_game.setdefault(str(game_id), {})
    for player in players:
        conf.setdefault(str(player), float(default_confidence))
    return conf


def normalize_belief_role(value: Any, valid_roles: List[str]) -> str:
    """Map free-form role text to a closed-set role label or 'unknown'."""
    if value is None:
        return "unknown"
    role_text = str(value).strip()
    if not role_text:
        return "unknown"
    if role_text.lower() == "unknown":
        return "unknown"
    mapped = extract_role_label(role_text, valid_roles)
    return mapped if mapped is not None else "unknown"


def normalize_belief_role_with_confidence(value: Any, valid_roles: List[str]) -> Tuple[str, float]:
    """Normalize role and derive a coarse confidence in [0,1]."""
    if isinstance(value, dict):
        role_raw = value.get("role")
        conf_raw = value.get("confidence")
        role = normalize_belief_role(role_raw, valid_roles)
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0 if role == "unknown" else 0.75
        conf = max(0.0, min(1.0, conf))
        if role == "unknown":
            conf = 0.0
        return role, conf

    role = normalize_belief_role(value, valid_roles)
    text = "" if value is None else str(value).strip()
    conf = 0.0 if role == "unknown" else 0.55
    if text:
        m = re.search(r"(?:confidence|conf)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, flags=re.IGNORECASE)
        if m:
            try:
                conf = float(m.group(1))
            except Exception:
                pass
    conf = max(0.0, min(1.0, conf))
    return role, conf


def apply_belief_updates(
    beliefs: Dict[str, str],
    updates: Dict[str, Any],
    valid_roles: List[str],
    belief_confidence: Dict[str, float] | None = None,
) -> Dict[str, str]:
    """Apply and normalize model belief updates into existing belief state."""
    for player, role_value in updates.items():
        role, conf = normalize_belief_role_with_confidence(role_value, valid_roles)
        beliefs[str(player)] = role
        if belief_confidence is not None:
            belief_confidence[str(player)] = conf
    return beliefs


def build_belief_vector_prompt(
    game_id: str,
    players: List[str],
    belief_vector: Dict[str, str],
    past_state_lines: List[str],
    current_quest: int,
    current_transcript: str,
    valid_roles: List[str],
) -> str:
    """Build a closed-set belief-update prompt for the current quest."""
    allowed_roles: List[str] = []
    seen: set[str] = set()
    for role in (valid_roles or []):
        role_label = str(role).strip()
        if not role_label:
            continue
        key = role_label.lower()
        if key in seen:
            continue
        seen.add(key)
        allowed_roles.append(role_label)
    if "unknown" not in seen:
        allowed_roles.append("unknown")

    roles_str = ", ".join(allowed_roles) if allowed_roles else "unknown"
    lines: List[str] = []
    lines.append("System: You are tracking hidden roles in an Avalon game.")
    lines.append(
        "Beliefs b = [b1, ..., b6] represent your current guess for each player's hidden role. "
        "Each belief bi must be EXACTLY one of the allowed role strings listed below (case-sensitive)."
    )
    lines.append("Allowed roles (closed set):")
    for role in allowed_roles:
        lines.append(f"- {role}")
    lines.append("")
    lines.append(f"Game id: {game_id}")
    lines.append("")
    lines.append("Global state so far (completed quests):")
    if past_state_lines:
        lines.extend(past_state_lines)
    else:
        lines.append("(no completed quests yet)")
    lines.append("")
    lines.append("Current beliefs b:")
    for player in players:
        role = belief_vector.get(player, "unknown")
        lines.append(f"- {player}: {role}")
    lines.append("")
    lines.append(f"Current quest: {current_quest}")
    lines.append("Quest transcript (only this round's chat):")
    lines.append(current_transcript or "(no player messages)")
    lines.append("")
    lines.append(
        "Task: Update the belief vector b_next after seeing this quest. "
        "Return an updated belief for every player in the same order, using only the roles {"
        + roles_str
        + "}."
    )

    example_pairs: List[str] = []
    if players:
        example_roles = allowed_roles if allowed_roles else ["unknown"]
        for i, player in enumerate(players[:2]):
            example_pairs.append(f'"{player}": "{example_roles[min(i, len(example_roles) - 1)]}"')
    else:
        example_pairs = ['"player-1": "unknown"', '"player-2": "unknown"']

    lines.append(
        "Respond ONLY with a single JSON object of the form: "
        "{\"beliefs\": {" + ", ".join(example_pairs) + ", ...}}."
    )
    lines.append("Output must be exactly one line of valid JSON.")
    lines.append("Do not include explanations, prefixes, markdown/code fences, trailing commas, or extra keys.")
    lines.append("If unsure about a player, set that player value to 'unknown' (still include every player).")
    return "\n".join(lines)
