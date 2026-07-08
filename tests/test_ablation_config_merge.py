from pathlib import Path

from scripts.run_ablation import apply_agent_overlay, load_yaml


def test_ablation_configs_are_mergeable():
    root = Path(__file__).resolve().parents[1]
    base = load_yaml(str(root / "configs" / "stocksim_tradingagents_ashare_long_horizon.yaml"))
    overlay = load_yaml(str(root / "configs" / "ablations" / "ashare_A1_no_macro.yaml"))
    merged = apply_agent_overlay(base, overlay)

    params = merged["agents"]["LongHorizon_TradingAgents"]["parameters"]
    assert params["long_horizon"]["enable_macro_regime"] is False
    assert params["experiment"]["run_id"] == "ashare_A1_no_macro"
    assert "simulation" in merged
