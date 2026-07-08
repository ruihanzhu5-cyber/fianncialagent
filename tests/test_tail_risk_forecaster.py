from tradingagents.risk.tail_risk_forecaster import TailRiskForecaster
from tradingagents.trading_intent import TradingIntent


def _intent(weight=0.5):
    return TradingIntent(
        action="BUY",
        target_weight=weight,
        max_trade_weight=None,
        confidence=0.8,
        rationale="test",
        rationale_type=None,
        valid_until=None,
        raw_decision="test",
        instrument="600519.SH",
    )


def test_high_cvar_risk_decays_or_vetoes():
    returns = [-0.08] * 20 + [0.001] * 120
    forecaster = TailRiskForecaster(risk_budget=0.01, min_sample_size=30, max_single_name_weight=1.0)
    decision = forecaster.veto_or_decay(_intent(0.8), returns)
    assert decision.veto_mask
    assert decision.adjusted_target_weight < 0.8


def test_low_sample_uses_fallback():
    forecaster = TailRiskForecaster(min_sample_size=30)
    estimate = forecaster.estimate([0.01, -0.02], 0.5)
    assert estimate.method == "vol_scaled_low_sample_fallback"
    assert estimate.sample_size == 2


def test_low_risk_passes():
    returns = [0.001, -0.001, 0.0005, -0.0002] * 40
    forecaster = TailRiskForecaster(risk_budget=0.2, min_sample_size=30)
    decision = forecaster.veto_or_decay(_intent(0.2), returns)
    assert not decision.veto_mask
    assert decision.adjusted_target_weight == 0.2
