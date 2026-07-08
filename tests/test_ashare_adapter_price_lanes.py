import pandas as pd

from tradingagents.dataflows.ashare_adapter import AShareDataAdapter


def test_prev_close_raw_is_emitted_by_ashare_adapter(monkeypatch):
    adapter = AShareDataAdapter.__new__(AShareDataAdapter)

    lanes = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-14", "2024-03-15"]),
        "open_raw": [10.0, 11.0],
        "high_raw": [10.5, 11.0],
        "low_raw": [9.8, 11.0],
        "close_raw": [10.0, 11.0],
        "volume_raw": [1000, 1200],
        "open_adjusted": [9.5, 10.5],
        "high_adjusted": [10.0, 10.5],
        "low_adjusted": [9.3, 10.5],
        "close_adjusted": [9.6, 10.5],
        "volume_adjusted": [1000, 1200],
        "return_close": [None, 0.09375],
    })
    monkeypatch.setattr(adapter, "load_price_lanes", lambda *args, **kwargs: lanes.copy())

    rows = adapter.load_aggregates("600519.SH", start_date="2024-03-14", end_date="2024-03-15", adjusted=True)

    assert rows[0]["raw_execution_prev_close"] is None
    assert rows[1]["raw_execution_prev_close"] == 10.0
    assert rows[1]["adjusted_prev_close"] == 9.6
