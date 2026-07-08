"""Temporal decoupling controller for event-triggered multi-agent activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ActivationDecision:
    full_debate_triggered: bool
    trigger_reasons: list[str]
    reuse_previous_intent: bool


class TemporalDecouplingController:
    def __init__(
        self,
        policy: str = "event_triggered",
        force_full_debate_every_n_trading_days: int = 20,
        return_vol_multiplier: float = 2.5,
        drawdown_delta_threshold: float = 0.04,
    ) -> None:
        self.policy = policy
        self.force_full_debate_every_n_trading_days = force_full_debate_every_n_trading_days
        self.return_vol_multiplier = return_vol_multiplier
        self.drawdown_delta_threshold = drawdown_delta_threshold
        self._last_full_debate_day: dict[str, datetime] = {}
        self._last_drawdown: dict[str, float] = {}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TemporalDecouplingController":
        raw = raw or {}
        return cls(
            policy=raw.get("policy", "event_triggered"),
            force_full_debate_every_n_trading_days=int(raw.get("force_full_debate_every_n_trading_days", 20)),
        )

    def decide(self, instrument: str, trade_date: str | datetime, compact_state: dict[str, Any]) -> ActivationDecision:
        if self.policy == "daily_full":
            self._mark(instrument, trade_date)
            return ActivationDecision(True, ["policy_daily_full"], False)
        reasons: list[str] = []
        dt = _to_dt(trade_date)
        last = self._last_full_debate_day.get(instrument)
        if last is None:
            reasons.append("initial_full_debate")
        elif (dt - last).days >= self.force_full_debate_every_n_trading_days:
            reasons.append("forced_periodic_full_debate")
        ret_1d = abs(float(compact_state.get("return_1d") or 0.0))
        vol = abs(float(compact_state.get("realized_vol_20d") or 0.0))
        if vol > 0 and ret_1d > self.return_vol_multiplier * vol:
            reasons.append("large_return_vs_vol")
        drawdown = float(compact_state.get("drawdown_from_60d_high") or 0.0)
        prev_dd = self._last_drawdown.get(instrument)
        if prev_dd is not None and abs(drawdown - prev_dd) > self.drawdown_delta_threshold:
            reasons.append("drawdown_delta")
        self._last_drawdown[instrument] = drawdown
        if compact_state.get("limit_hit_state") in {"LIMIT_UP", "LIMIT_DOWN", "ONE_WORD_LIMIT_UP", "ONE_WORD_LIMIT_DOWN"}:
            reasons.append("limit_hit_cluster_detected")
        if compact_state.get("material_news_available"):
            reasons.append("material_news_available")
        triggered = bool(reasons)
        if triggered:
            self._mark(instrument, dt)
        return ActivationDecision(triggered, reasons, not triggered)

    def _mark(self, instrument: str, trade_date: str | datetime) -> None:
        self._last_full_debate_day[instrument] = _to_dt(trade_date)


def _to_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)[:10])
