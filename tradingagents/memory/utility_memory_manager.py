"""Utility-aware memory pruning for long-horizon trading runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryItem:
    event_time: str
    ticker: str
    regime: str
    action: str
    target_weight: float
    realized_return_5d: float | None = None
    realized_return_20d: float | None = None
    realized_return_60d: float | None = None
    max_drawdown_after_decision: float | None = None
    veto_triggered: bool = False
    was_black_swan: bool = False
    utility_score: float = 0.0
    text_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class UtilityMemoryManager:
    def __init__(
        self,
        memory_budget_tokens: int = 1200,
        crisis_memory_pin: bool = True,
        crisis_pin_quarters: int = 4,
        crisis_drawdown_threshold: float = -0.12,
    ) -> None:
        self.memory_budget_tokens = memory_budget_tokens
        self.crisis_memory_pin = crisis_memory_pin
        self.crisis_pin_quarters = crisis_pin_quarters
        self.crisis_drawdown_threshold = crisis_drawdown_threshold
        self.items: list[MemoryItem] = []
        self.last_trace: dict[str, Any] = {}

    def add(self, item: MemoryItem) -> None:
        if item.utility_score == 0.0:
            item.utility_score = self._tail_utility(item)
        self.items.append(item)

    def retrieve(
        self,
        ticker: str,
        current_regime: str,
        as_of: str | None = None,
        max_items: int | None = None,
    ) -> list[MemoryItem]:
        selected: list[MemoryItem] = []
        candidates = [m for m in self.items if m.ticker == ticker]
        scores = {id(m): self._score(m, current_regime, selected, as_of) for m in candidates}
        budget = self.memory_budget_tokens
        cap = max_items or len(candidates)
        pinned = [m for m in candidates if self._is_pinned(m, as_of)]
        for item in sorted(pinned, key=lambda m: scores[id(m)], reverse=True):
            if item not in selected:
                selected.append(item)
                budget -= self._token_estimate(item)
        remaining = [m for m in candidates if m not in selected]
        for item in sorted(remaining, key=lambda m: scores[id(m)], reverse=True):
            if len(selected) >= cap:
                break
            cost = self._token_estimate(item)
            if cost <= budget:
                selected.append(item)
                budget -= cost
        self.last_trace = {
            "memory_items_used": len(selected),
            "memory_scores": {m.event_time: round(scores[id(m)], 6) for m in candidates},
            "memory_pruned_count": max(0, len(candidates) - len(selected)),
        }
        return selected

    def _score(
        self,
        item: MemoryItem,
        current_regime: str,
        selected: list[MemoryItem],
        as_of: str | None,
    ) -> float:
        recency = self._recency(item, as_of)
        tail = self._tail_utility(item)
        reuse = abs(item.utility_score)
        regime = 1.0 if item.regime == current_regime else 0.2
        redundancy = max((self._similarity(item.text_summary, s.text_summary) for s in selected), default=0.0)
        pin_bonus = 2.0 if self._is_pinned(item, as_of) else 0.0
        return 0.30 * recency + 0.30 * tail + 0.20 * reuse + 0.20 * regime - 0.25 * redundancy + pin_bonus

    def _is_pinned(self, item: MemoryItem, as_of: str | None) -> bool:
        if not self.crisis_memory_pin:
            return False
        crisis = item.was_black_swan or (
            item.max_drawdown_after_decision is not None
            and item.max_drawdown_after_decision < self.crisis_drawdown_threshold
        )
        if not crisis:
            return False
        if not as_of:
            return True
        age_days = max(0, (_parse_dt(as_of) - _parse_dt(item.event_time)).days)
        return age_days <= self.crisis_pin_quarters * 92

    def _tail_utility(self, item: MemoryItem) -> float:
        dd = abs(item.max_drawdown_after_decision or 0.0)
        returns = [abs(x or 0.0) for x in (item.realized_return_5d, item.realized_return_20d, item.realized_return_60d)]
        return min(1.0, dd + max(returns, default=0.0) + (0.2 if item.veto_triggered else 0.0))

    def _recency(self, item: MemoryItem, as_of: str | None) -> float:
        if not as_of:
            return 1.0
        age_days = max(0, (_parse_dt(as_of) - _parse_dt(item.event_time)).days)
        return 1.0 / (1.0 + age_days / 90.0)

    def _token_estimate(self, item: MemoryItem) -> int:
        return max(1, len(item.text_summary) // 4 + 24)

    def _similarity(self, a: str, b: str) -> float:
        left = set(a.lower().split())
        right = set(b.lower().split())
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value[:10])
