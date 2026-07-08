from tradingagents.trading_intent import TradingIntentParser


def test_json_buy_intent():
    intent = TradingIntentParser().parse(
        '{"action":"BUY","target_weight":0.3,"confidence":0.8,"rationale":"ok"}',
        current_weight=0.1,
        instrument="600519.SH",
    )
    assert intent.action == "BUY"
    assert intent.target_weight == 0.3
    assert intent.confidence == 0.8


def test_sell_defaults_to_zero_not_add_position():
    intent = TradingIntentParser().parse("FINAL TRANSACTION PROPOSAL: SELL", current_weight=0.4)
    assert intent.action == "SELL"
    assert intent.target_weight == 0.0


def test_hold_defaults_to_current_weight_not_clear_position():
    intent = TradingIntentParser().parse("FINAL TRANSACTION PROPOSAL: HOLD", current_weight=0.37)
    assert intent.action == "HOLD"
    assert intent.target_weight == 0.37


def test_exit_defaults_to_zero():
    intent = TradingIntentParser().parse("```json\n{\"action\":\"EXIT\",\"confidence\":0.6}\n```", current_weight=0.5)
    assert intent.action == "EXIT"
    assert intent.target_weight == 0.0


def test_parse_failure_is_conservative_hold():
    intent = TradingIntentParser().parse('{"action":"BUY",', current_weight=0.2)
    assert intent.action == "HOLD"
    assert intent.target_weight == 0.2
    assert intent.parse_error is not None
