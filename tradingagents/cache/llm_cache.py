"""Deterministic file cache helpers for LLM calls."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


class LLMCallCache:
    def __init__(self, cache_dir: str = "cache/llm_calls") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def key(self, *, model_name: str, prompt_template_version: str, structured_state: Any, tool_outputs: Any, config_flags: Any) -> str:
        payload = {
            "model_name": model_name,
            "prompt_template_version": prompt_template_version,
            "structured_state": structured_state,
            "tool_outputs": tool_outputs,
            "config_flags": config_flags,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: dict[str, Any]) -> str:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        return path

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")
