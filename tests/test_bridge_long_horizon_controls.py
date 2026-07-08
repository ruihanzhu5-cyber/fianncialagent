import asyncio
import importlib
import sys
import types
from collections import defaultdict
from datetime import datetime

import pytest


class _DummyMetrics:
    def __init__(self):
        self.portfolio_time_series = []
        self.trade_history = []
        self._returns = []


class _DummyTraderAgent:
    def __init__(self, *args, **kwargs):
        self.cash = kwargs.get("initial_cash", 100000.0)
        self.portfolio_value = self.cash
        self.long_qty = defaultdict(int, kwargs.get("initial_positions", {}))
        self.short_qty = defaultdict(int)
        self.prices = defaultdict(float)
        self.metrics = _DummyMetrics()
        self.current_time = datetime(2024, 3, 15)
        self.current_tick_id = 1
        self.logger = types.SimpleNamespace(error=lambda *a, **k: None, warning=lambda *a, **k: None)
        self.placed_orders = []

    async def place_order(self, instrument, side, quantity, order_type, **kwargs):
        order_id = f"order-{len(self.placed_orders) + 1}"
        self.placed_orders.append((order_id, instrument, side, quantity, order_type, kwargs))
        return order_id

    async def on_trade_execution(self, trade_data):
        if trade_data["role"] == "BUYER":
            self.long_qty[trade_data["instrument"]] += int(trade_data["quantity"])
            self.cash -= float(trade_data["price"]) * int(trade_data["quantity"])
        else:
            self.long_qty[trade_data["instrument"]] -= int(trade_data["quantity"])
            self.cash += float(trade_data["price"]) * int(trade_data["quantity"])
        self.metrics.trade_history.append({"action": "BUY" if trade_data["role"] == "BUYER" else "SELL"})


class _DummyGraph:
    def __init__(self, *args, **kwargs):
        self.memory_log = types.SimpleNamespace(store_decision=lambda **kwargs: None)

    def propagate(self, instrument, trade_date):
        return {"final_trade_decision": '{"action":"HOLD","target_weight":0.0}'}, "HOLD"


@pytest.fixture()
def bridge_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", types.ModuleType("agents"))
    monkeypatch.setitem(sys.modules, "agents.benchmark_traders", types.ModuleType("agents.benchmark_traders"))
    trader_mod = types.ModuleType("agents.benchmark_traders.trader")
    trader_mod.TraderAgent = _DummyTraderAgent
    monkeypatch.setitem(sys.modules, "agents.benchmark_traders.trader", trader_mod)
    messages_mod = types.ModuleType("utils.messages")
    messages_mod.MessageType = types.SimpleNamespace(DECISION_RESPONSE="DECISION_RESPONSE")
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.messages", messages_mod)
    orders_mod = types.ModuleType("utils.orders")
    orders_mod.Side = types.SimpleNamespace(BUY=types.SimpleNamespace(value="BUY"), SELL=types.SimpleNamespace(value="SELL"))
    monkeypatch.setitem(sys.modules, "utils.orders", orders_mod)
    graph_mod = types.ModuleType("tradingagents.graph.trading_graph")
    graph_mod.TradingAgentsGraph = _DummyGraph
    monkeypatch.setitem(sys.modules, "tradingagents.graph.trading_graph", graph_mod)
    sys.modules.pop("bridge_simulation", None)
    return importlib.import_module("bridge_simulation")


def _agent(bridge_module, long_horizon):
    return bridge_module.TradingAgentsStockSimAgent(
        instrument_exchange_map={"600519.SH": "ex"},
        initial_cash=100000.0,
        tradingagents_config={"market_region": "cn", "data_source": "akshare"},
        data_source_config={"600519.SH": {"data_source": "akshare"}},
        long_horizon=long_horizon,
        china_microstructure={"enabled": True},
    )


def test_macro_disabled_does_not_call_macro_agent(bridge_module):
    agent = _agent(bridge_module, {"enable_macro_regime": False})
    called = {"value": False}
    agent.macro_agent.infer = lambda **kwargs: called.__setitem__("value", True)
    state = agent._build_compact_state("600519.SH", {"data": {"close": 10}})
    if agent.macro_enabled and agent._needs_regime_refresh("600519.SH"):
        agent.macro_agent.infer(instrument="600519.SH", timestamp=agent.current_time, compact_state=state)
    assert not called["value"]


def test_dynamic_memory_on_uses_utility_manager(bridge_module):
    agent = _agent(bridge_module, {"enable_dynamic_memory": True})
    agent.utility_memory.add(bridge_module.MemoryItem(event_time="2024-01-01", ticker="600519.SH", regime="neutral", action="BUY", target_weight=0.2, text_summary="useful"))
    memory = agent._retrieve_memory("600519.SH", "neutral")
    assert memory["dynamic_memory_enabled"] is True
    assert memory["memory_items_used"] == 1


def test_dynamic_memory_off_uses_fixed_recent_memory(bridge_module):
    agent = _agent(bridge_module, {"enable_dynamic_memory": False})
    agent.memory_pool.append({"instrument": "600519.SH", "action": "BUY"})
    memory = agent._retrieve_memory("600519.SH", "neutral")
    assert memory["dynamic_memory_enabled"] is False
    assert memory["memory_items_used"] == 1


def test_risk_metric_drawdown_routes_to_risk_veto_agent(bridge_module):
    agent = _agent(bridge_module, {"enable_tail_risk_veto": True, "risk_metric": "drawdown_threshold"})
    assert agent.risk_metric == "drawdown_threshold"
    assert agent.risk_agent is not None


def test_transaction_cost_changes_accounting(bridge_module):
    agent = _agent(bridge_module, {})
    agent.pending_microstructure["order-1"] = {
        "instrument": "600519.SH",
        "total_transaction_cost": 1.3,
        "rounded_qty": 100,
        "trade_date": "2024-03-15",
    }
    asyncio.run(agent.on_trade_execution({
        "order_id": "order-1",
        "instrument": "600519.SH",
        "role": "BUYER",
        "quantity": 100,
        "price": 10,
    }))
    assert agent.cash == pytest.approx(98998.7)
    assert agent.microstructure_guard.ledger.get("600519.SH").frozen_qty == 100
