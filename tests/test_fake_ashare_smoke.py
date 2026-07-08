from pathlib import Path

import pandas as pd

from tradingagents.dataflows.ashare_market_realism import ChinaMicrostructureGuard, MicrostructureConfig


def test_fake_ashare_fixture_exercises_limit_and_t_plus_one():
    fixture = Path(__file__).parent / "fixtures" / "fake_ashare_ohlcv.csv"
    df = pd.read_csv(fixture)
    limit_up = df.iloc[1]
    guard = ChinaMicrostructureGuard(MicrostructureConfig())
    blocked_buy = guard.prepare_order(
        instrument="600519.SH",
        side="BUY",
        raw_qty=100,
        held_qty=0,
        trade_date=limit_up.date,
        candle={
            "raw_execution_open": limit_up.raw_open,
            "raw_execution_high": limit_up.raw_high,
            "raw_execution_low": limit_up.raw_low,
            "raw_execution_close": limit_up.raw_close,
            "raw_execution_prev_close": df.iloc[0].raw_close,
        },
    )
    assert blocked_buy.block_reason == "ONE_WORD_LIMIT_UP"

    guard.record_filled("600519.SH", "BUY", 100, "2024-03-14")
    same_day_sell = guard.prepare_order(
        instrument="600519.SH",
        side="SELL",
        raw_qty=100,
        held_qty=100,
        trade_date="2024-03-14",
        candle={"raw_execution_close": 10.1, "raw_execution_prev_close": 9.9},
    )
    assert same_day_sell.block_reason == "T_PLUS_ONE"
