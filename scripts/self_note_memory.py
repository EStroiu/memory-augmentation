from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from scripts.template_summary import MemoryEntry, quest_state_line
from scripts.json_utils import extract_object_from_brace, extract_role_label, typechat_repair_to_json


def parse_role_note_json(raw_text: Optional[str]) -> Dict[str, Optional[str]]:
    """Best-effort parse for {'role': ..., 'memory_note': ...} style outputs."""
    out: Dict[str, Optional[str]] = {"role_raw": None, "memory_note": None}
    if not isinstance(raw_text, str) or not raw_text.strip():
        return out
    s = raw_text.strip()

    def _extract_from_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        role_val = obj.get("role")
        if isinstance(role_val, str):
            out["role_raw"] = role_val
        for key in ["memory_note", "note", "summary", "memory", "running_note"]:
            note_val = obj.get(key)
            if isinstance(note_val, str):
                out["memory_note"] = note_val
                break

    try:
        obj = json.loads(s)
        _extract_from_obj(obj)
        if out["role_raw"] is not None or out["memory_note"] is not None:
            return out
    except Exception:
        pass

    # Fall back to scanning balanced JSON objects from free-form outputs.
    try:
        for i, ch in enumerate(s):
            if ch != "{":
                continue
            obj = extract_object_from_brace(s, i)
            _extract_from_obj(obj)
            if out["role_raw"] is not None or out["memory_note"] is not None:
                return out
    except Exception:
        pass
    return out


def normalize_memory_note(note: Optional[str], max_chars: int = 320) -> Optional[str]:
    """Keep the self-authored memory compact and single-line friendly."""
    if not isinstance(note, str):
        return None
    compact = " ".join(note.split())
    if not compact:
        return None
    if len(compact) <= int(max_chars):
        return compact
    # Keep a clean sentence-like tail when truncating.
    truncated = compact[: int(max_chars)].rstrip(" ,.;:")
    return truncated + "..."


def build_llm_self_note_prompt(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Build prompt where the model maintains a compact running note per game."""
    del retrieved  # Not needed in this strategy; kept for strategy interface parity.

    game_id: str = target.game_id
    quests: Dict[int, List[Dict]] = ctx["game_quests"].get(game_id, {})
    players: List[str] = ctx["game_players"].get(game_id, [])
    valid_roles: List[str] = ctx["valid_roles"]
    role_list = ", ".join(valid_roles) if valid_roles else "unknown"

    rolling_note_by_game: Dict[str, str] = ctx.setdefault("rolling_note_by_game", {})
    prev_note = str(rolling_note_by_game.get(game_id, "")).strip()
    if not prev_note:
        prev_note = "(none yet)"

    past_state_lines: List[str] = []
    for q in sorted(quests.keys()):
        if int(q) >= int(target.quest):
            continue
        past_state_lines.append(quest_state_line(game_id, int(q), quests[q]))

    context_block = "\n".join(past_state_lines[-4:]) if past_state_lines else "(no completed quests yet)"

    lines: List[str] = []
    lines.append("System: You are an Avalon role analyst maintaining a compact running memory.")
    lines.append("")
    lines.append(f"Game: {game_id} | Quest: {int(target.quest)}")
    if players:
        lines.append("Players: " + ", ".join(players))
    lines.append("")
    lines.append("Running memory note from previous quests:")
    lines.append(prev_note)
    lines.append("")
    lines.append("Recent completed-quest state (most recent first omitted, chronological shown):")
    lines.append(context_block)
    lines.append("")
    lines.append("Current quest structured context:")
    lines.append(target.text)
    lines.append("")
    lines.append("Task:")
    lines.append(f"1) Predict the role of the player who proposed the party. Valid roles: {role_list}.")
    lines.append("2) Write an updated running note for future quests.")
    lines.append("")
    lines.append("Constraints for updated note:")
    lines.append("- Max 320 characters.")
    lines.append("- Do not repeat full transcript details.")
    lines.append("")
    lines.append(
        "Respond ONLY with one-line JSON matching exactly this schema: "
        '{"role":"<ROLE>","memory_note":"<UPDATED_NOTE>"}. '
        "No markdown, no prose, no extra keys."
    )

    prompt = "\n".join(lines)
    meta: Dict[str, Any] = {
        "mode": "llm_self_note",
        "self_note_enabled": True,
        "self_note_prev_chars": len(prev_note) if prev_note != "(none yet)" else 0,
    }
    return prompt, meta


def postprocess_typechat_role_note(
    raw_pred: Optional[str],
    valid_roles: List[str],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse role + running memory note with optional JSON repair."""

    openai_model: str = ctx.get("openai_model", "")
    openai_api_key: Optional[str] = ctx.get("openai_api_key")
    llm_trace: Optional[List[Dict[str, Any]]] = ctx.get("llm_trace")

    parsed = parse_role_note_json(raw_pred)
    pred_role = extract_role_label(parsed.get("role_raw"), valid_roles)
    memory_note_next = normalize_memory_note(parsed.get("memory_note"))
    used_repair = False
    repair_attempted = False
    repaired_preview: Optional[str] = None

    text = raw_pred or ""
    needs_repair = pred_role is None or memory_note_next is None
    if needs_repair and text:
        schema_hint = '{"role": "<ROLE>", "memory_note": "<SHORT_NOTE_MAX_320_CHARS>"}'
        repaired = typechat_repair_to_json(
            text,
            schema_hint,
            openai_model,
            openai_api_key,
            True,
            trace=llm_trace,
        )
        repair_attempted = True
        if isinstance(repaired, str):
            repaired_preview = repaired.replace("\n", " ")[:160]
        repaired_parsed = parse_role_note_json(repaired)
        rep_role = extract_role_label(repaired_parsed.get("role_raw"), valid_roles)
        rep_note = normalize_memory_note(repaired_parsed.get("memory_note"))
        if rep_role is not None:
            pred_role = rep_role
            used_repair = True
        if rep_note is not None:
            memory_note_next = rep_note
            used_repair = True

    fallback_reason: Optional[str] = None
    if pred_role is None:
        if not text.strip():
            fallback_reason = "empty_raw_prediction"
        else:
            fallback_reason = "role_unparsed"

    return {
        "pred_role": pred_role,
        "beliefs_obj": None,
        "used_repair": used_repair,
        "repair_attempted": repair_attempted,
        "repaired_preview": repaired_preview,
        "proposer_from_beliefs": None,
        "beliefs_preview": None,
        "fallback_reason": fallback_reason,
        "decision_path": [
            "role_note:raw" if parsed.get("role_raw") or parsed.get("memory_note") else "role_note:none",
            "role_note:repair" if used_repair else "role_note:no_repair",
        ],
        "memory_note_next": memory_note_next,
        "memory_note_next_chars": len(memory_note_next) if isinstance(memory_note_next, str) else 0,
    }
