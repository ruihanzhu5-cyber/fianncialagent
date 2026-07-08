"""JSONL decision trace exporter."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DecisionTraceExporter:
    def __init__(self, run_id: str, output_dir: str = "traces", enabled: bool = True) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.enabled = enabled
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, f"{run_id}_decision_trace.jsonl")

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"run_id": self.run_id, **record}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)
