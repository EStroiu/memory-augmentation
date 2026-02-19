#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from scripts.template_summary import MemoryEntry
from scripts.belief_vector import build_belief_vector_prompt

# ---------- Basic prompt assembly ----------

def assemble_prompt(target: MemoryEntry, retrieved: List[Tuple[MemoryEntry, float]]) -> str:
    parts: List[str] = []
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
    parts: List[str] = []
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


def assemble_belief_baseline_prompt(
    game_id: str,
    players: List[str],
    belief_vector: Dict[str, str],
    past_state_lines: List[str],
    current_quest: int,
    current_transcript: str,
    valid_roles: List[str],
) -> str:
    return build_belief_vector_prompt(
        game_id=game_id,
        players=players,
        belief_vector=belief_vector,
        past_state_lines=past_state_lines,
        current_quest=current_quest,
        current_transcript=current_transcript,
        valid_roles=valid_roles,
    )


def prompt_stats(prompt_with_memory: str, prompt_baseline: str) -> Dict[str, Dict[str, int]]:
    def _s(t: str) -> Dict[str, int]:
        return {
            "chars": len(t),
            "words": len(t.split()),
            "lines": t.count("\n") + (1 if t else 0),
        }

    return {"with_memory": _s(prompt_with_memory), "baseline": _s(prompt_baseline)}


def assemble_prompt_with_meta(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    task_suffix: str,
) -> Tuple[str, Dict[str, Any]]:
    """Assemble full memory-augmented prompt and attach simple metadata.

    The name reflects that we no longer enforce any explicit token budget.
    """
    base = assemble_prompt(target, retrieved) + task_suffix
    meta: Dict[str, Any] = {
        "used_memory": len(retrieved),
        "total_memory_available": len(retrieved),
    }
    return base, meta


def assemble_baseline_prompt_with_meta(
    target: MemoryEntry,
    task_suffix: str,
) -> Tuple[str, Dict[str, Any]]:
    """Assemble full baseline prompt and attach simple metadata."""
    base = assemble_baseline_prompt(target) + task_suffix
    meta: Dict[str, Any] = {}
    return base, meta
