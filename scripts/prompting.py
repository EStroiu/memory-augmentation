#!/usr/bin/env python3
"""Prompt assembly, budgeting, and stats utilities.

This module centralizes prompt-related logic so that memory construction and
retrieval stay in `memory_utils.py` and evaluation scripts can stay compact.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from scripts.memory_utils import MemoryEntry, heuristic_summary


# ---------- Token estimation ----------

_token_regex = re.compile(r"\w+|[^\s\w]")


def estimate_tokens(text: str) -> int:
    """Rough token estimate compatible with LLaMA2-length budgeting.

    This avoids heavy tokenizer dependencies. Empirically adequate for enforcing a
    4k limit: overestimates slightly for safety. Each word-like sequence or punctuation
    chunk counts as a token.
    """
    if not text:
        return 0
    return len(_token_regex.findall(text))


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


def prompt_stats(prompt_with_memory: str, prompt_baseline: str) -> Dict[str, Dict[str, int]]:
    def _s(t: str) -> Dict[str, int]:
        return {
            "chars": len(t),
            "words": len(t.split()),
            "lines": t.count("\n") + (1 if t else 0),
            "tokens": estimate_tokens(t),
        }

    return {"with_memory": _s(prompt_with_memory), "baseline": _s(prompt_baseline)}


# ---------- Adaptive compression helpers ----------


def _compress_entry_text(entry_text: str, max_transcript_lines: int = 10) -> str:
    """Aggressively compress an entry by retaining header + first N transcript lines.

    Assumes original format produced by build_template_entry_for_quest.
    """
    lines = entry_text.splitlines()
    out: List[str] = []
    transcript_started = False
    kept_transcript = 0
    for ln in lines:
        if ln.startswith("Transcript (player messages only):"):
            out.append(ln)
            transcript_started = True
            continue
        if not transcript_started:
            out.append(ln)
        else:
            if ln.strip() == "":
                continue
            if ln.startswith("[Turn "):
                if kept_transcript < max_transcript_lines:
                    out.append(ln)
                    kept_transcript += 1
            # ignore remaining transcript lines beyond cap
    # Add heuristic summary for context after compression
    summary = heuristic_summary("\n".join(lines))
    out.append(summary.strip())
    return "\n".join(out)


def _entry_header_only(entry_text: str) -> str:
    lines = entry_text.splitlines()
    out: List[str] = []
    for ln in lines:
        if ln.startswith("Transcript (player messages only):"):
            break
        out.append(ln)
    out.append("(transcript omitted)")
    return "\n".join(out)


# ---------- Budgeted prompt assembly ----------


def assemble_prompt_budgeted(
    target: MemoryEntry,
    retrieved: List[Tuple[MemoryEntry, float]],
    max_tokens: int,
    task_suffix: str,
    safety_buffer: int = 50,
) -> Tuple[str, Dict[str, Any]]:
    """Build a memory-augmented prompt that respects a token budget.

    Strategies applied in order until under budget:
    1. Incremental inclusion of memory entries (stop before exceeding budget).
    2. If last added entry causes overflow, attempt compressed version of that entry.
    3. If still overflowing, revert that entry entirely and stop.
    4. If over budget even with zero memory entries, compress target transcript.
    5. If still over, use header-only target.
    Returns prompt and metadata describing compression/truncation actions.
    """
    meta: Dict[str, Any] = {
        "budget": int(max_tokens),
        "estimated_tokens": 0,
        "used_memory": 0,
        "total_memory_available": len(retrieved),
        "actions": [],
        "truncated": False,
    }
    parts: List[str] = []
    parts.append("System: You are helping analyze an Avalon game round. Use the memory context to inform your reasoning.")
    parts.append("")
    parts.append(f"Target: GAME {target.game_id} | QUEST {target.quest}")
    parts.append("")
    parts.append("Memory context (top-k similar rounds):")
    base_prefix = "\n".join(parts) + "\n"
    current = base_prefix

    # Add memory entries one by one
    for i, (entry, score) in enumerate(retrieved, start=1):
        block_lines = [f"---- Memory {i} (cosine sim ~ {score:.3f}) | {entry.entry_id}", entry.text, ""]
        tentative = current + "\n".join(block_lines)
        est = estimate_tokens(tentative + task_suffix)
        if est <= (max_tokens - safety_buffer):
            current = tentative
            meta["used_memory"] = i
            continue
        # Try compressed version of this entry
        compressed_text = _compress_entry_text(entry.text, max_transcript_lines=8)
        block_lines_comp = [f"---- Memory {i} (compressed, cos sim ~ {score:.3f}) | {entry.entry_id}", compressed_text, ""]
        tentative_comp = current + "\n".join(block_lines_comp)
        est_comp = estimate_tokens(tentative_comp + task_suffix)
        if est_comp <= (max_tokens - safety_buffer):
            current = tentative_comp
            meta["used_memory"] = i
            meta["actions"].append({"entry": entry.entry_id, "action": "compressed"})
            continue
        # Revert including this entry entirely and stop adding more
        meta["actions"].append({"entry": entry.entry_id, "action": "skipped"})
        meta["truncated"] = True
        break

    current += "Your task: Provide an analysis or next-step reasoning for the target round using the context above.\nAnswer:" + task_suffix
    est_final = estimate_tokens(current)

    # If still over budget without any memory entries attempt target compression
    if est_final > max_tokens:
        meta["truncated"] = True
        target_header = _entry_header_only(target.text)
        target_compressed = _compress_entry_text(target.text, max_transcript_lines=10)
        add_block = "\n\nTarget round context (compressed):\n" + target_compressed + "\n"
        tentative_target_comp = base_prefix + add_block + "Memory context omitted due to budget.\nAnswer:" + task_suffix
        est_target_comp = estimate_tokens(tentative_target_comp)
        if est_target_comp <= max_tokens:
            current = tentative_target_comp
            meta["actions"].append({"target": target.entry_id, "action": "target_compressed"})
            est_final = est_target_comp
        else:
            fallback = base_prefix + "\nTarget round (header only):\n" + target_header + "\nAnswer:" + task_suffix
            est_fallback = estimate_tokens(fallback)
            current = fallback
            meta["actions"].append({"target": target.entry_id, "action": "target_header_only"})
            est_final = est_fallback

    meta["estimated_tokens"] = est_final
    return current, meta


def assemble_baseline_prompt_budgeted(
    target: MemoryEntry,
    max_tokens: int,
    task_suffix: str,
    safety_buffer: int = 50,
) -> Tuple[str, Dict[str, Any]]:
    """Budgeted baseline prompt containing only target round context.

    Compression order: full -> compressed -> header-only.
    """
    meta: Dict[str, Any] = {
        "budget": int(max_tokens),
        "estimated_tokens": 0,
        "actions": [],
        "truncated": False,
    }
    parts = [
        "System: You are helping analyze an Avalon game round. No external memory is provided for this baseline.",
        "",
        f"Target: GAME {target.game_id} | QUEST {target.quest}",
        "",
        "Target round context:",
        target.text,
        "",
        "Answer:" + task_suffix,
    ]
    prompt_full = "\n".join(parts)
    est_full = estimate_tokens(prompt_full)
    if est_full <= max_tokens - safety_buffer:
        meta["estimated_tokens"] = est_full
        return prompt_full, meta

    # Try compressed
    meta["truncated"] = True
    compressed = _compress_entry_text(target.text, max_transcript_lines=12)
    parts[5] = compressed
    prompt_comp = "\n".join(parts)
    est_comp = estimate_tokens(prompt_comp)
    if est_comp <= max_tokens:
        meta["actions"].append({"target": target.entry_id, "action": "compressed"})
        meta["estimated_tokens"] = est_comp
        return prompt_comp, meta

    # Header only fallback
    header_only = _entry_header_only(target.text)
    parts[5] = header_only
    prompt_hdr = "\n".join(parts)
    est_hdr = estimate_tokens(prompt_hdr)
    meta["actions"].append({"target": target.entry_id, "action": "header_only"})
    meta["estimated_tokens"] = est_hdr
    return prompt_hdr, meta
