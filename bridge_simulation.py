"""Bridge StockSim's event loop to TradingAgents' multi-agent decision graph.

This module keeps both upstream projects intact:
- StockSim owns the simulation clock, exchange, accounting, and metrics.
- TradingAgents owns analyst debate, risk debate, and final trade decision.

The bridge is intentionally thin and training-free. Long-horizon extensions are
implemented as cached context and an execution-time risk gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from agents.benchmark_traders.trader import TraderAgent
    from utils.messages import MessageType
    from utils.orders import Side
except ImportError:  # Allows static import from the repository root.
    TraderAgent = object  # type: ignore[assignment,misc]
    MessageType = None  # type: ignore[assignment]
    Side = None  # type: ignore[assignment]

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


@dataclass
class RegimeState:
    as_of: str
    horizon: str
    regime: str
    risk_budget: float
    constraints: list[str]
    summary: str


@dataclass
class TradingIntent:
    instrument: str
    action: str
    target_weight: float
    confidence: float
    rationale: str
    raw_decision: str


class LongHorizonMemoryPool:
    """Compact memory ledger used before prompts reach TradingAgents."""

    def __init__(self, max_items: int = 80) -> None:
        self.max_items = max_items
        self._items: list[dict[str, Any]] = []

    def retrieve(self, instrument: str) -> dict[str, Any]:
        relevant = [m for m in self._items if m.get("instrument") == instrument]
        return {"decision_ledger": relevant[-self.max_items :]}

    def append(self, item: dict[str, Any]) -> None:
        self._items.append(item)
        self._items = self._items[-self.max_items :]


class MacroRegimeAgent:
    """Training-free coarse-cycle macro node.

    Replace the heuristic body with a GPT/Claude JSON call for experiments.
    """

    def infer(self, instrument: str, timestamp: datetime, compact_state: dict[str, Any]) -> RegimeState:
        # CHANGED FOR LONG-HORIZON:
        # TODO(long-horizon): call a weekly/monthly Macro Agent here. It should
        # summarize market regime, capital budget, sector bias, and hard risk
        # constraints, then cache the output for downstream micro decisions.
        vol = compact_state.get("realized_vol_20d") or 0.0
        drawdown = compact_state.get("drawdown_from_60d_high") or 0.0
        regime = "risk_off" if vol > 0.035 or drawdown < -0.12 else "neutral"
        risk_budget = 0.35 if regime == "risk_off" else 0.75
        return RegimeState(
            as_of=timestamp.isoformat(),
            horizon="1M-1Q",
            regime=regime,
            risk_budget=risk_budget,
            constraints=["cap single-name target_weight by regime risk_budget"],
            summary=f"{instrument}: {regime}; vol={vol:.4f}; drawdown={drawdown:.4f}",
        )


class RiskVetoAgent:
    """Execution-time long-horizon risk gate."""

    def __init__(self, max_drawdown_threshold: float = 0.18) -> None:
        self.max_drawdown_threshold = max_drawdown_threshold

    def veto_or_adjust(
        self,
        intent: TradingIntent,
        regime_state: RegimeState | None,
        compact_state: dict[str, Any],
    ) -> tuple[TradingIntent, dict[str, Any]]:
        # CHANGED FOR LONG-HORIZON:
        # TODO(long-horizon): replace this deterministic guard with a Risk Agent
        # that estimates 1-2 quarter MDD/CVaR from StockSim history and vetoes
        # short-horizon aggressive actions before order submission.
        recent_dd = abs(float(compact_state.get("drawdown_from_60d_high") or 0.0))
        budget = regime_state.risk_budget if regime_state else 1.0
        proposed_weight = abs(intent.target_weight)
        veto = recent_dd > self.max_drawdown_threshold or proposed_weight > budget

        if not veto:
            return intent, {
                "veto": False,
                "estimated_drawdown": recent_dd,
                "risk_budget": budget,
                "reason": "within long-horizon risk gate",
            }

        adjusted_weight = min(intent.target_weight, budget) if intent.target_weight > 0 else 0.0
        adjusted = TradingIntent(
            instrument=intent.instrument,
            action="HOLD" if adjusted_weight == 0 else intent.action,
            target_weight=adjusted_weight,
            confidence=intent.confidence,
            rationale=f"{intent.rationale}\n[RISK_VETO] recent_dd={recent_dd:.4f}, budget={budget:.4f}",
            raw_decision=intent.raw_decision,
        )
        return adjusted, {
            "veto": True,
            "estimated_drawdown": recent_dd,
            "risk_budget": budget,
            "reason": "long-horizon drawdown or budget threshold breached",
        }


class TradingAgentsStockSimAgent(TraderAgent):  # type: ignore[misc,valid-type]
    """StockSim trader whose decision brain is TradingAgentsGraph.propagate()."""

    def __init__(
        self,
        instrument_exchange_map: dict[str, str],
        agent_id: str | None = None,
        rabbitmq_host: str = "localhost",
        tradingagents_config: dict[str, Any] | None = None,
        selected_analysts: list[str] | None = None,
        macro_cycle: str = "month",
        max_drawdown_threshold: float = 0.18,
        min_trade_notional: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **kwargs,
        )
        config = dict(DEFAULT_CONFIG)
        config.update(tradingagents_config or {})
        self.trading_graph = TradingAgentsGraph(
            selected_analysts=tuple(selected_analysts or ["market", "news", "fundamentals"]),
            debug=bool(config.get("debug", False)),
            config=config,
        )
        self.macro_cycle = macro_cycle
        self.min_trade_notional = min_trade_notional
        self.memory_pool = LongHorizonMemoryPool()
        self.macro_agent = MacroRegimeAgent()
        self.risk_agent = RiskVetoAgent(max_drawdown_threshold=max_drawdown_threshold)
        self.regime_cache: dict[str, RegimeState] = {}

    async def on_market_data_update(self, instrument: str, snapshot: dict[str, Any]) -> None:
        data = snapshot.get("data", {}) or {}
        if not data:
            await self._publish_decision_done()
            return

        compact_state = self._build_compact_state(instrument, snapshot)

        # CHANGED FOR LONG-HORIZON:
        # Macro/Micro decoupling: only refresh expensive coarse-cycle context at
        # month/week boundary, then inject it into every micro trading decision.
        if self._needs_regime_refresh(instrument):
            self.regime_cache[instrument] = self.macro_agent.infer(
                instrument=instrument,
                timestamp=self.current_time,
                compact_state=compact_state,
            )

        regime_state = self.regime_cache.get(instrument)
        self._inject_long_horizon_context(instrument, compact_state, regime_state)

        try:
            trade_date = self.current_time.strftime("%Y-%m-%d")
            final_state, processed_signal = self.trading_graph.propagate(instrument, trade_date)
            intent = self._parse_intent(instrument, final_state, processed_signal)

            # CHANGED FOR LONG-HORIZON:
            # Risk Agent veto runs after TradingAgents debate, before StockSim execution.
            safe_intent, risk_record = self.risk_agent.veto_or_adjust(
                intent=intent,
                regime_state=regime_state,
                compact_state=compact_state,
            )
            await self._execute_target_weight(safe_intent, data)
            self._record_bridge_decision(instrument, compact_state, safe_intent, risk_record)
        except Exception as exc:
            self.logger.error("TradingAgents bridge decision failed for %s: %s", instrument, exc)
        finally:
            await self._publish_decision_done()

    def _build_compact_state(self, instrument: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        data = snapshot.get("data", {}) or {}
        indicators = snapshot.get("indicators", {}) or {}
        close = float(data.get("close") or self.prices.get(instrument) or 0.0)
        self.prices[instrument] = close

        price_points = [
            p.get("value")
            for p in self.metrics.portfolio_time_series[-60:]
            if isinstance(p, dict) and p.get("value") is not None
        ]
        dd = 0.0
        if price_points:
            peak = max(price_points)
            current = price_points[-1]
            dd = current / peak - 1.0 if peak else 0.0

        return {
            "timestamp": self.current_time.isoformat() if self.current_time else None,
            "instrument": instrument,
            "ohlcv": {
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "close": close,
                "volume": data.get("volume"),
            },
            "indicators": indicators,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "long_qty": self.long_qty[instrument],
            "short_qty": self.short_qty[instrument],
            "drawdown_from_60d_high": dd,
            "realized_vol_20d": indicators.get("volatility_20d") or indicators.get("atr"),
            "memory": self.memory_pool.retrieve(instrument),
        }

    def _inject_long_horizon_context(
        self,
        instrument: str,
        compact_state: dict[str, Any],
        regime_state: RegimeState | None,
    ) -> None:
        memory_context = compact_state.get("memory", {})
        regime_text = json.dumps(regime_state.__dict__ if regime_state else {}, ensure_ascii=False)
        compact_text = json.dumps(compact_state, ensure_ascii=False, default=str)
        self.trading_graph.memory_log.store_decision(
            ticker=instrument,
            trade_date=f"bridge_context_{self.current_time.strftime('%Y-%m-%d')}",
            final_trade_decision=(
                "Long-horizon external context for this StockSim tick.\n"
                f"regime_state={regime_text}\n"
                f"compact_state={compact_text}\n"
                f"memory={json.dumps(memory_context, ensure_ascii=False, default=str)}"
            ),
        )

    def _needs_regime_refresh(self, instrument: str) -> bool:
        cached = self.regime_cache.get(instrument)
        if cached is None or self.current_time is None:
            return True
        cached_time = datetime.fromisoformat(cached.as_of)
        if self.macro_cycle == "week":
            return self.current_time.isocalendar()[:2] != cached_time.isocalendar()[:2]
        return (self.current_time.year, self.current_time.month) != (
            cached_time.year,
            cached_time.month,
        )

    def _parse_intent(
        self,
        instrument: str,
        final_state: dict[str, Any],
        processed_signal: str,
    ) -> TradingIntent:
        raw_decision = str(final_state.get("final_trade_decision", processed_signal))
        action = self._normalize_action(processed_signal or raw_decision)
        target_weight = self._extract_target_weight(raw_decision, action)
        confidence = self._extract_confidence(raw_decision)
        return TradingIntent(
            instrument=instrument,
            action=action,
            target_weight=target_weight,
            confidence=confidence,
            rationale=raw_decision[:2000],
            raw_decision=raw_decision,
        )

    async def _execute_target_weight(self, intent: TradingIntent, data: dict[str, Any]) -> None:
        price = float(data.get("close") or data.get("open") or 0.0)
        if price <= 0 or intent.action == "HOLD":
            return

        target_value = self.portfolio_value * max(0.0, min(abs(intent.target_weight), 1.0))
        current_value = self.long_qty[intent.instrument] * price
        delta_value = target_value - current_value
        if abs(delta_value) < self.min_trade_notional:
            return

        quantity = int(abs(delta_value) // price)
        if quantity <= 0:
            return

        if delta_value > 0 and intent.action in {"BUY", "STRONG_BUY"}:
            await self.place_order(
                intent.instrument,
                Side.BUY.value,
                quantity,
                "MARKET",
                explanation=intent.rationale,
            )
        elif delta_value < 0 or intent.action in {"SELL", "STRONG_SELL"}:
            await self.place_order(
                intent.instrument,
                Side.SELL.value,
                quantity,
                "MARKET",
                explanation=intent.rationale,
            )

    def _record_bridge_decision(
        self,
        instrument: str,
        compact_state: dict[str, Any],
        intent: TradingIntent,
        risk_record: dict[str, Any],
    ) -> None:
        self.memory_pool.append({
            "timestamp": compact_state.get("timestamp"),
            "instrument": instrument,
            "action": intent.action,
            "target_weight": intent.target_weight,
            "confidence": intent.confidence,
            "risk": risk_record,
        })

    async def _publish_decision_done(self) -> None:
        if MessageType is None:
            return
        await self.publish_time(
            msg_type=MessageType.DECISION_RESPONSE,
            payload={"tick_id": self.current_tick_id},
            routing_key="simulation_clock",
        )

    @staticmethod
    def _normalize_action(text: str) -> str:
        upper = text.upper()
        if "STRONG SELL" in upper:
            return "STRONG_SELL"
        if "SELL" in upper:
            return "SELL"
        if "STRONG BUY" in upper:
            return "STRONG_BUY"
        if "BUY" in upper:
            return "BUY"
        return "HOLD"

    @staticmethod
    def _extract_target_weight(text: str, action: str) -> float:
        patterns = [
            r"target[_\s-]*weight\D+([0-9]+(?:\.[0-9]+)?)\s*%",
            r"target[_\s-]*weight\D+([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s+)?portfolio",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value / 100.0 if value > 1 else value
        if action in {"STRONG_BUY", "STRONG_SELL"}:
            return 0.50
        if action in {"BUY", "SELL"}:
            return 0.25
        return 0.0

    @staticmethod
    def _extract_confidence(text: str) -> float:
        match = re.search(r"confidence\D+([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        if not match:
            return 0.5
        value = float(match.group(1))
        return value / 100.0 if value > 1 else value

