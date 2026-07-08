"""Run a StockSim config with an ablation overlay."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_agent_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for agent in merged.get("agents", {}).values():
        if agent.get("type") != "TradingAgentsStockSimAgent":
            continue
        params = agent.setdefault("parameters", {})
        for section in ("experiment", "china_microstructure", "long_horizon", "multi_agent_activation"):
            if section in overlay:
                params[section] = deep_merge(params.get(section, {}), overlay[section])
    return merged


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = apply_agent_overlay(load_yaml(args.base), load_yaml(args.overlay))
    output = args.output
    if not output:
        fd, output = tempfile.mkstemp(prefix="stocksim_ablation_", suffix=".yaml")
        os.close(fd)
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    if args.dry_run:
        print(output)
        return 0
    return subprocess.call([sys.executable, "third_party/StockSim/main_launcher.py", output])


if __name__ == "__main__":
    raise SystemExit(main())
