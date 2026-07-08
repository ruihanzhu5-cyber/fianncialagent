from tradingagents.activation.temporal_decoupling import TemporalDecouplingController


def test_initial_day_triggers_full_debate_then_reuses():
    controller = TemporalDecouplingController(force_full_debate_every_n_trading_days=20)
    first = controller.decide("600519.SH", "2024-01-02", {})
    second = controller.decide("600519.SH", "2024-01-03", {})
    assert first.full_debate_triggered
    assert "initial_full_debate" in first.trigger_reasons
    assert not second.full_debate_triggered
    assert second.reuse_previous_intent


def test_large_return_triggers_full_debate():
    controller = TemporalDecouplingController()
    controller.decide("600519.SH", "2024-01-02", {})
    decision = controller.decide("600519.SH", "2024-01-03", {"return_1d": 0.08, "realized_vol_20d": 0.01})
    assert decision.full_debate_triggered
    assert "large_return_vs_vol" in decision.trigger_reasons


def test_daily_full_policy_always_triggers():
    controller = TemporalDecouplingController(policy="daily_full")
    assert controller.decide("600519.SH", "2024-01-02", {}).full_debate_triggered
    assert controller.decide("600519.SH", "2024-01-03", {}).full_debate_triggered
