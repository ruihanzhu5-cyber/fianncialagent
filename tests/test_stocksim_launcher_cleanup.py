from pathlib import Path


def test_stocksim_launcher_registers_only_tradingagents_llm():
    launcher = Path("third_party/StockSim/main_launcher.py").read_text(encoding="utf-8")

    assert '"TradingAgentsStockSimAgent": TradingAgentsStockSimAgent' in launcher
    assert "LLMTradingAgent" not in launcher
    assert '"Buy_And_Hold_Trader": BuyAndHoldTrader' in launcher
    assert '"SMA_Trader": SMATrader' in launcher
    assert '"MACD_Trader": MACDTrader' in launcher
