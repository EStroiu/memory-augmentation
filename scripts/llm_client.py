#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, Optional
import requests


def llm_role_predict(prompt: str, use_llm: bool, model_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call an LLM to get a raw prediction string.

    Returns a dict with keys: prediction (raw text), error (optional), provider.
    This is intentionally minimal; higher-level code is responsible for parsing
    and interpreting the prediction.
    """
    if not use_llm:
        return {
            "prediction": None,
            "cost": 0.0,
            "tokens": 0,
            "provider": None,
            "model": model_name,
            "duration_s": 0.0,
        }

    if not model_name:
        return {
            "prediction": None,
            "error": "No model_name provided",
            "provider": None,
            "model": model_name,
            "duration_s": 0.0,
        }

    # Ollama provider
    if model_name.lower().startswith("ollama:"):
        if requests is None:
            return {"prediction": None, "error": "requests not installed; pip install requests", "provider": "ollama"}
        model = model_name.split(":", 1)[1]
        base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        url = base.rstrip("/") + "/api/generate"
        t0 = time.time()
        try:
            resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or data.get("message") or ""
            return {
                "prediction": text,
                "provider": "ollama",
                "model": model_name,
                "duration_s": float(time.time() - t0),
            }
        except Exception as e:  # pragma: no cover - network failure path
            return {
                "prediction": None,
                "error": f"ollama request failed: {e}",
                "provider": "ollama",
                "model": model_name,
                "duration_s": float(time.time() - t0),
            }

    # OpenAI (placeholder; extend if desired)
    if model_name.lower().startswith("openai:"):
        t0 = time.time()
        try:
            import openai  # type: ignore  # noqa: F401
        except Exception:  # pragma: no cover - optional dependency
            return {
                "prediction": None,
                "error": "openai library not installed",
                "provider": "openai",
                "model": model_name,
                "duration_s": float(time.time() - t0),
            }
        if not api_key:
            return {
                "prediction": None,
                "error": "OPENAI_API_KEY not provided",
                "provider": "openai",
                "model": model_name,
                "duration_s": float(time.time() - t0),
            }
        # No-op placeholder: avoid making external calls by default
        return {
            "prediction": None,
            "note": "openai path stubbed; implement as needed",
            "provider": "openai",
            "model": model_name,
            "duration_s": float(time.time() - t0),
        }

    return {
        "prediction": None,
        "error": "Unknown LLM provider; use model_name starting with 'ollama:' or 'openai:'",
        "provider": None,
        "model": model_name,
        "duration_s": 0.0,
    }
