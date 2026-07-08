from tradingagents.dataflows.ashare_market_realism import (
    BoardLotSizer,
    ChinaMicrostructureGuard,
    ChinaPriceLimitGuard,
    MicrostructureConfig,
    SellableInventoryLedger,
)


def test_t_plus_one_blocks_same_day_sell_and_allows_next_day():
    ledger = SellableInventoryLedger()
    ledger.record_buy("600519.SH", 100, "2024-03-15")
    assert ledger.max_sellable("600519.SH", 100, "2024-03-15") == 0
    assert ledger.max_sellable("600519.SH", 100, "2024-03-18") == 100


def test_microstructure_guard_blocks_same_day_sell_with_trace_reason():
    guard = ChinaMicrostructureGuard(MicrostructureConfig())
    guard.record_filled("600519.SH", "BUY", 100, "2024-03-15")
    decision = guard.prepare_order(
        instrument="600519.SH",
        side="SELL",
        raw_qty=100,
        held_qty=100,
        trade_date="2024-03-15",
        candle={"raw_execution_close": 100, "open": 100, "high": 100, "low": 100, "prev_close": 99},
    )
    assert not decision.executable
    assert decision.block_reason == "T_PLUS_ONE"


def test_one_word_price_limits_block_buy_and_sell():
    limit_guard = ChinaPriceLimitGuard()
    can_buy, reason_buy = limit_guard.evaluate(
        side="BUY",
        symbol="600519.SH",
        open_price=110,
        high_price=110,
        low_price=110,
        close_price=110,
        prev_close=100,
    )
    can_sell, reason_sell = limit_guard.evaluate(
        side="SELL",
        symbol="600519.SH",
        open_price=90,
        high_price=90,
        low_price=90,
        close_price=90,
        prev_close=100,
    )
    assert not can_buy and reason_buy == "ONE_WORD_LIMIT_UP"
    assert not can_sell and reason_sell == "ONE_WORD_LIMIT_DOWN"


def test_board_lot_sizing_and_odd_lot_full_exit():
    sizer = BoardLotSizer()
    assert sizer.size("BUY", 146) == 100
    assert sizer.size("SELL", 146, held_qty=146, full_exit=False) == 100
    assert sizer.size("SELL", 146, held_qty=146, full_exit=True) == 146


def test_transaction_costs_include_sell_stamp_duty():
    guard = ChinaMicrostructureGuard(MicrostructureConfig(commission_bps=3, stamp_duty_bps_sell=5, slippage_bps=5))
    buy = guard.calculate_cost("BUY", 100, 10)
    sell = guard.calculate_cost("SELL", 100, 10)
    assert buy.commission == 0.3
    assert buy.stamp_duty == 0
    assert buy.slippage == 0.5
    assert sell.commission == 0.3
    assert sell.stamp_duty == 0.5
    assert sell.slippage == 0.5


def test_submitted_unfilled_order_does_not_change_t_plus_one_inventory():
    guard = ChinaMicrostructureGuard(MicrostructureConfig())
    decision = guard.prepare_order(
        instrument="600519.SH",
        side="SELL",
        raw_qty=100,
        held_qty=100,
        trade_date="2024-03-15",
        candle={"raw_execution_close": 100, "raw_execution_prev_close": 99},
    )
    assert not decision.executable
    assert decision.block_reason == "T_PLUS_ONE"


def test_microstructure_component_flags_can_be_disabled():
    guard = ChinaMicrostructureGuard(MicrostructureConfig(enforce_board_lot=False))
    buy = guard.prepare_order(
        instrument="600519.SH",
        side="BUY",
        raw_qty=146,
        held_qty=0,
        trade_date="2024-03-15",
        candle={"raw_execution_close": 10, "raw_execution_prev_close": 9.8},
    )
    assert buy.executable
    assert buy.rounded_qty == 146

    guard = ChinaMicrostructureGuard(MicrostructureConfig(enforce_t_plus_one=False))
    sell = guard.prepare_order(
        instrument="600519.SH",
        side="SELL",
        raw_qty=100,
        held_qty=100,
        trade_date="2024-03-15",
        candle={"raw_execution_close": 10, "raw_execution_prev_close": 9.8},
    )
    assert sell.executable

    guard = ChinaMicrostructureGuard(MicrostructureConfig(enforce_price_limit=False))
    limit_buy = guard.prepare_order(
        instrument="600519.SH",
        side="BUY",
        raw_qty=100,
        held_qty=0,
        trade_date="2024-03-15",
        candle={
            "raw_execution_open": 11,
            "raw_execution_high": 11,
            "raw_execution_low": 11,
            "raw_execution_close": 11,
            "raw_execution_prev_close": 10,
        },
    )
    assert limit_buy.executable


def test_price_limit_blocks_with_adapter_style_fields():
    guard = ChinaMicrostructureGuard(MicrostructureConfig())
    decision = guard.prepare_order(
        instrument="600519.SH",
        side="BUY",
        raw_qty=100,
        held_qty=0,
        trade_date="2024-03-15",
        candle={
            "raw_execution_open": 11,
            "raw_execution_high": 11,
            "raw_execution_low": 11,
            "raw_execution_close": 11,
            "raw_execution_prev_close": 10,
        },
    )
    assert not decision.executable
    assert decision.limit_hit_state == "ONE_WORD_LIMIT_UP"
