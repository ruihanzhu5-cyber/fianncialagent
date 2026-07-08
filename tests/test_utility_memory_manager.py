from tradingagents.memory.utility_memory_manager import MemoryItem, UtilityMemoryManager


def test_crisis_memory_is_retained_over_plain_old_memory():
    manager = UtilityMemoryManager(memory_budget_tokens=80)
    manager.add(MemoryItem(event_time="2023-08-01", ticker="600519.SH", regime="risk_off", action="SELL", target_weight=0, was_black_swan=True, text_summary="crisis drawdown liquidity shock"))
    manager.add(MemoryItem(event_time="2024-01-01", ticker="600519.SH", regime="neutral", action="BUY", target_weight=0.2, text_summary="ordinary calm period"))
    selected = manager.retrieve("600519.SH", "neutral", as_of="2024-03-01", max_items=1)
    assert selected[0].was_black_swan
    assert manager.last_trace["memory_pruned_count"] == 1


def test_redundant_memory_can_be_pruned_by_budget():
    manager = UtilityMemoryManager(memory_budget_tokens=50)
    manager.add(MemoryItem(event_time="2024-01-01", ticker="600519.SH", regime="neutral", action="BUY", target_weight=0.2, text_summary="same same same neutral buy"))
    manager.add(MemoryItem(event_time="2024-01-02", ticker="600519.SH", regime="neutral", action="BUY", target_weight=0.2, text_summary="same same same neutral buy"))
    selected = manager.retrieve("600519.SH", "neutral", as_of="2024-02-01")
    assert len(selected) == 1
