from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from scripts.llm_client import llm_role_predict


def parse_json_role(text: str | None) -> str | None:
    """Try to parse a JSON object like {"role": "..."} from text."""
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("role"), str):
            return obj.get("role")
    except Exception:
        pass
    try:
        m = re.search(r"\{[^{}]*\}", s, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("role"), str):
                return obj.get("role")
    except Exception:
        pass
    return None


def extract_object_from_brace(s: str, start_idx: int) -> Any:
    """Given a string and an index of a '{', return json.loads of the balanced object.

    Uses a simple brace counter to find the matching '}'. Returns None on failure.
    """
    depth = 0
    end_idx = -1
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx == -1:
        return None
    try:
        return json.loads(s[start_idx:end_idx])
    except Exception:
        return None


def extract_beliefs_object(text: str | None) -> Dict[str, Any] | None:
    """Extract a JSON object containing a top-level key "beliefs" from arbitrary text.

    Tries direct json.loads first; then scans for a balanced-brace JSON object
    that contains the key. Returns the parsed dict if it contains a dict at
    key "beliefs".
    """
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("beliefs"), dict):
            return obj
    except Exception:
        pass
    try:
        m = re.search(r"\bbeliefs\b", s, flags=re.IGNORECASE)
        candidate_indices: List[int] = []
        if m:
            left_brace_before = s.rfind("{", 0, m.start())
            if left_brace_before != -1:
                candidate_indices.append(left_brace_before)
            left_brace_after = s.find("{", m.start())
            if left_brace_after != -1:
                candidate_indices.append(left_brace_after)
        else:
            candidate_indices = [i for i, ch in enumerate(s) if ch == "{"]
        for idx in candidate_indices:
            obj = extract_object_from_brace(s, idx)
            if isinstance(obj, dict) and isinstance(obj.get("beliefs"), dict):
                return obj
    except Exception:
        return None
    return None


def typechat_repair_to_json(
    raw_text: str,
    schema_hint: str,
    model_name: str,
    api_key: str | None,
    enabled: bool,
) -> str | None:
    """One-shot repair call that asks the LLM to emit strict JSON only.

    schema_hint should describe the exact JSON to produce. Returns the model
    response text or None on failure.
    """
    if not enabled:
        return None
    try:
        repair_prompt = (
            "Convert the following answer into strict JSON only. "
            "Do not include any explanation or extra keys.\n"
            f"Schema: {schema_hint}\n"
            "Answer to convert:\n"
            "<<<\n" + raw_text + "\n>>>\n"
            "Output JSON only on a single line."
        )
        rep = llm_role_predict(repair_prompt, True, model_name, api_key)
        return rep.get("prediction")
    except Exception:
        return None


def extract_role_label(text: str | None, valid_roles: List[str]) -> str | None:
    """Parse TypeChat-like JSON first, then fall back to fuzzy label extraction.

    1) Parse JSON {"role": "<ROLE>"} and validate against valid_roles.
    2) Fallback: contains/word-boundary matching.
    """
    role = parse_json_role(text)
    if role and valid_roles:
        for vr in valid_roles:
            if role.strip().lower() == vr.lower():
                return vr
    if not text:
        return None
    t = text.strip()
    tl = t.lower()
    hits = [r for r in valid_roles if r.lower() in tl]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return sorted(hits, key=len, reverse=True)[0]
    for r in valid_roles:
        try:
            if re.search(rf"\b{re.escape(r)}\b", t, flags=re.IGNORECASE):
                return r
        except re.error:
            continue
    return None
