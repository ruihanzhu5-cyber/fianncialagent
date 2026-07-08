"""Bridge StockSim's event loop to TradingAgents' multi-agent decision graph.

This module keeps both upstream projects intact:
- StockSim owns the simulation clock, exchange, accounting, and metrics.
- TradingAgents owns analyst debate, risk debate, and final trade decision.

The bridge is intentionally thin and training-free. Long-horizon extensions are
implemented as cached context and an execution-time risk gate.
"""

from __future__ import annotations

import json
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

from tradingagents.dataflows.ashare_adapter import AShareDataAdapter
from tradingagents.dataflows.ashare_market_realism import (
    ChinaMicrostructureGuard,
    MicrostructureConfig,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.activation.temporal_decoupling import TemporalDecouplingController
from tradingagents.memory.utility_memory_manager import MemoryItem, UtilityMemoryManager
from tradingagents.risk.tail_risk_forecaster import TailRiskForecaster
from tradingagents.trading_intent import TradingIntent, TradingIntentParser
from tradingagents.trace.decision_trace import DecisionTraceExporter, config_hash


@dataclass
class RegimeState:
    as_of: str
    horizon: str
    regime: str
    max_position_weight: float
    cvar_budget: float
    constraints: list[str]
    summary: str

    @property
    def risk_budget(self) -> float:
        return self.max_position_weight


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
        max_position_weight = 0.35 if regime == "risk_off" else 0.75
        cvar_budget = 0.08 if regime == "risk_off" else 0.12
        return RegimeState(
            as_of=timestamp.isoformat(),
            horizon="1M-1Q",
            regime=regime,
            max_position_weight=max_position_weight,
            cvar_budget=cvar_budget,
            constraints=["cap single-name target_weight by regime max_position_weight"],
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
        budget = regime_state.max_position_weight if regime_state else 1.0
        proposed_weight = abs(intent.target_weight)
        veto = recent_dd > self.max_drawdown_threshold or proposed_weight > budget

        if not veto:
            return intent, {
                "veto": False,
                "veto_mask": False,
                "decay_applied": False,
                "risk_rewrite_reason": None,
                "estimated_drawdown": recent_dd,
                "max_position_weight": budget,
                "reason": "within long-horizon risk gate",
            }

        adjusted_weight = min(intent.target_weight, budget) if intent.target_weight > 0 else 0.0
        adjusted = TradingIntent(
            action="HOLD" if adjusted_weight == 0 else intent.action,
            target_weight=adjusted_weight,
            max_trade_weight=intent.max_trade_weight,
            confidence=intent.confidence,
            rationale=f"{intent.rationale}\n[RISK_VETO] recent_dd={recent_dd:.4f}, budget={budget:.4f}",
            rationale_type=intent.rationale_type,
            valid_until=intent.valid_until,
            raw_decision=intent.raw_decision,
            parse_error=intent.parse_error,
            instrument=intent.instrument,
        )
        return adjusted, {
            "veto": True,
            "veto_mask": True,
            "decay_applied": adjusted_weight != intent.target_weight,
            "risk_rewrite_reason": "long-horizon drawdown or max-position threshold breached",
            "estimated_drawdown": recent_dd,
            "max_position_weight": budget,
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
        data_source_config: dict[str, dict[str, str]] | None = None,
        selected_analysts: list[str] | None = None,
        macro_cycle: str = "month",
        max_drawdown_threshold: float = 0.18,
        min_trade_notional: float = 1000.0,
        experiment: dict[str, Any] | None = None,
        china_microstructure: dict[str, Any] | None = None,
        long_horizon: dict[str, Any] | None = None,
        multi_agent_activation: dict[str, Any] | None = None,
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
        self.data_source_config = data_source_config or {}
        if self._uses_ashare_data(config):
            # CHANGED FOR DOMESTIC DATA: route TradingAgents market/indicator/fundamental
            # calls through AkShare so StockSim and the LLM debate consume one A-share tape.
            data_vendors = dict(config.get("data_vendors", {}))
            data_vendors.update({
                "core_stock_apis": "akshare",
                "technical_indicators": "akshare",
                "fundamental_data": "akshare",
            })
            config["data_vendors"] = data_vendors
        self.trading_graph = TradingAgentsGraph(
            selected_analysts=tuple(selected_analysts or ["market", "news", "fundamentals"]),
            debug=bool(config.get("debug", False)),
            config=config,
        )
        self.macro_cycle = macro_cycle
        self.min_trade_notional = min_trade_notional
        self.experiment_config = experiment or {}
        self.long_horizon_config = long_horizon or {}
        self.activation_config = multi_agent_activation or {}
        self.macro_enabled = bool(self.long_horizon_config.get("enable_macro_regime", True))
        self.dynamic_memory_enabled = bool(self.long_horizon_config.get("enable_dynamic_memory", True))
        self.risk_veto_enabled = bool(self.long_horizon_config.get("enable_tail_risk_veto", True))
        self.risk_metric = str(self.long_horizon_config.get("risk_metric", "historical_cvar"))
        self.run_id = str(self.experiment_config.get("run_id", "ashare_lh_run"))
        self.config_hash = config_hash({
            "experiment": self.experiment_config,
            "china_microstructure": china_microstructure or {},
            "long_horizon": self.long_horizon_config,
            "multi_agent_activation": self.activation_config,
        })
        self.memory_pool = LongHorizonMemoryPool()
        self.utility_memory = UtilityMemoryManager(
            memory_budget_tokens=int(self.long_horizon_config.get("memory_budget_tokens", 1200)),
            crisis_memory_pin=bool(self.long_horizon_config.get("crisis_memory_pin", True)),
        )
        self.macro_agent = MacroRegimeAgent()
        self.risk_agent = RiskVetoAgent(max_drawdown_threshold=max_drawdown_threshold)
        self.intent_parser = TradingIntentParser()
        self.microstructure_guard = ChinaMicrostructureGuard(
            MicrostructureConfig.from_dict(china_microstructure)
        )
        self.tail_risk = TailRiskForecaster(
            alpha=float(self.long_horizon_config.get("cvar_alpha", 0.95)),
            horizon_days=int(self.long_horizon_config.get("cvar_horizon_days", 60)),
            max_single_name_weight=float(self.long_horizon_config.get("max_single_name_weight", 0.35)),
        )
        self.activation_controller = TemporalDecouplingController.from_dict(self.activation_config)
        self.trace_exporter = DecisionTraceExporter(
            run_id=self.run_id,
            enabled=bool(self.experiment_config.get("export_decision_trace", True)),
        )
        self.previous_intents: dict[str, TradingIntent] = {}
        self.pending_microstructure: dict[str, dict[str, Any]] = {}
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
        if self.macro_enabled and self._needs_regime_refresh(instrument):
            self.regime_cache[instrument] = self.macro_agent.infer(
                instrument=instrument,
                timestamp=self.current_time,
                compact_state=compact_state,
            )

        regime_state = self.regime_cache.get(instrument) if self.macro_enabled else None
        if regime_state is not None:
            compact_state["regime"] = regime_state.regime
        compact_state["memory"] = self._retrieve_memory(
            instrument, regime_state.regime if regime_state else "unknown"
        )
        self._inject_long_horizon_context(instrument, compact_state, regime_state)

        try:
            trade_date = self.current_time.strftime("%Y-%m-%d")
            activation = self.activation_controller.decide(instrument, trade_date, compact_state)
            if activation.full_debate_triggered or instrument not in self.previous_intents:
                final_state, processed_signal = self.trading_graph.propagate(instrument, trade_date)
                intent = self._parse_intent(instrument, final_state, processed_signal)
                self.previous_intents[instrument] = intent
            else:
                final_state, processed_signal = {}, ""
                intent = self.previous_intents[instrument]

            # CHANGED FOR LONG-HORIZON:
            # Risk Agent veto runs after TradingAgents debate, before StockSim execution.
            if not self.risk_veto_enabled:
                safe_intent = intent
                risk_record = self._no_risk_record(regime_state)
            elif self.risk_metric == "historical_cvar":
                if regime_state:
                    self.tail_risk.max_single_name_weight = regime_state.max_position_weight
                risk_decision = self.tail_risk.veto_or_decay(
                    intent=intent,
                    returns=list(getattr(self.metrics, "_returns", []))[-252:],
                    risk_budget=regime_state.cvar_budget if regime_state else None,
                )
                safe_intent = risk_decision.intent
                risk_record = {
                    "veto": risk_decision.veto_mask,
                    "veto_mask": risk_decision.veto_mask,
                    "veto_reason": risk_decision.veto_reason,
                    "decay_applied": risk_decision.decay_applied,
                    "risk_rewrite_reason": risk_decision.risk_rewrite_reason,
                    "risk_engine": "historical_cvar",
                    "risk_metric": self.risk_metric,
                    "cvar_95_60d": risk_decision.estimate.cvar_95_60d,
                    "var_95_60d": risk_decision.estimate.var_95_60d,
                    "expected_mdd_60d": risk_decision.estimate.expected_mdd_60d,
                    "risk_sample_size": risk_decision.estimate.sample_size,
                    "low_sample_fallback": risk_decision.low_sample_fallback,
                }
            else:
                safe_intent, risk_record = self.risk_agent.veto_or_adjust(
                    intent=intent,
                    regime_state=regime_state,
                    compact_state=compact_state,
                )
                risk_record.update({"risk_engine": "drawdown_threshold", "risk_metric": self.risk_metric})
            execution_record = await self._execute_target_weight(safe_intent, data)
            self._record_bridge_decision(
                instrument,
                compact_state,
                safe_intent,
                risk_record,
                activation.__dict__,
                execution_record,
                raw_intent=intent,
            )
        except Exception as exc:
            self.logger.error("TradingAgents bridge decision failed for %s: %s", instrument, exc)
        finally:
            await self._publish_decision_done()

    def _build_compact_state(self, instrument: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        data = snapshot.get("data", {}) or {}
        indicators = snapshot.get("indicators", {}) or {}
        close = float(data.get("close") or self.prices.get(instrument) or 0.0)
        adjusted_close = data.get("adjusted_close")
        raw_execution_price = float(data.get("raw_execution_close") or data.get("close") or close)
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
        try:
            ashare_symbol = AShareDataAdapter.to_exchange_symbol(instrument)
        except ValueError:
            ashare_symbol = instrument

        return {
            "timestamp": self.current_time.isoformat() if self.current_time else None,
            "instrument": instrument,
            # CHANGED FOR DOMESTIC DATA: expose A-share normalization and qfq/calendar provenance
            # to downstream prompts, memory records, and risk-veto audit logs.
            "ashare_symbol": ashare_symbol,
            "price_adjustment": "qfq" if self._instrument_uses_ashare(instrument) else None,
            "analysis_price_lane": "qfq_adjusted_price" if self._instrument_uses_ashare(instrument) else None,
            "execution_price_lane": "raw_execution_price" if self._instrument_uses_ashare(instrument) else None,
            "return_price_lane": "adjusted_return_series" if self._instrument_uses_ashare(instrument) else None,
            "adjusted_price": adjusted_close,
            "raw_execution_price": raw_execution_price,
            "return_1d": data.get("return_close"),
            "used_price_lane": data.get("used_price_lane"),
            "trade_calendar": (
                "akshare.tool_trade_date_hist_sina"
                if self._instrument_uses_ashare(instrument)
                else None
            ),
            "ohlcv": {
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "close": close,
                "volume": data.get("volume"),
                "adjusted_close": adjusted_close,
                "raw_execution_close": raw_execution_price,
            },
            "indicators": indicators,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "long_qty": self.long_qty[instrument],
            "short_qty": self.short_qty[instrument],
            "drawdown_from_60d_high": dd,
            "realized_vol_20d": indicators.get("volatility_20d") or indicators.get("atr"),
            "memory": self._retrieve_memory(instrument, "unknown"),
        }

    def _retrieve_memory(self, instrument: str, regime: str | None) -> dict[str, Any]:
        if not self.dynamic_memory_enabled:
            recent = self.memory_pool.retrieve(instrument)
            return {
                **recent,
                "dynamic_memory_enabled": False,
                "memory_items_used": len(recent.get("decision_ledger", [])),
                "memory_pruned_count": 0,
                "memory_scores": {},
                "crisis_pinned_count": 0,
            }
        as_of = self.current_time.strftime("%Y-%m-%d") if self.current_time else None
        selected = self.utility_memory.retrieve(
            ticker=instrument,
            current_regime=regime or "unknown",
            as_of=as_of,
        )
        trace = dict(self.utility_memory.last_trace)
        crisis_pinned_count = sum(
            1 for item in selected if item.was_black_swan or (item.max_drawdown_after_decision or 0.0) < -0.12
        )
        return {
            "dynamic_memory_enabled": True,
            "decision_ledger": [item.__dict__ for item in selected],
            "crisis_pinned_count": crisis_pinned_count,
            **trace,
        }

    def _uses_ashare_data(self, config: dict[str, Any]) -> bool:
        if config.get("market_region") == "cn" or config.get("data_source") == "akshare":
            return True
        return any(
            (source_cfg or {}).get("data_source") == "akshare"
            for source_cfg in self.data_source_config.values()
        )

    def _instrument_uses_ashare(self, instrument: str) -> bool:
        source_cfg = self.data_source_config.get(instrument, {})
        return source_cfg.get("data_source") == "akshare"

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
        price = float(self.prices.get(instrument) or 0.0)
        current_weight = (
            (self.long_qty[instrument] * price) / self.portfolio_value
            if self.portfolio_value and price > 0
            else 0.0
        )
        return self.intent_parser.parse(raw_decision or processed_signal, current_weight, instrument)

    async def _execute_target_weight(self, intent: TradingIntent, data: dict[str, Any]) -> dict[str, Any]:
        instrument = intent.instrument or ""
        price = float(data.get("raw_execution_close") or data.get("close") or data.get("open") or 0.0)
        base_record: dict[str, Any] = {
            "order_executable": False,
            "microstructure_block_reason": None,
            "used_price_lane": "raw_execution_price" if data.get("raw_execution_close") else "default_price",
            "commission": 0.0,
            "stamp_duty": 0.0,
            "slippage": 0.0,
        }
        if price <= 0 or intent.action == "HOLD" or not instrument:
            return base_record

        target_value = self.portfolio_value * max(0.0, min(abs(intent.target_weight), 1.0))
        current_value = self.long_qty[instrument] * price
        delta_value = target_value - current_value
        if abs(delta_value) < self.min_trade_notional:
            return base_record

        quantity = int(abs(delta_value) // price)
        if quantity <= 0:
            return base_record

        if delta_value > 0 and intent.action == "BUY":
            decision = self.microstructure_guard.prepare_order(
                instrument=instrument,
                side="BUY",
                raw_qty=quantity,
                held_qty=self.long_qty[instrument],
                trade_date=self.current_time,
                candle=data,
            )
            if decision.executable:
                order_id = await self.place_order(
                    instrument,
                    Side.BUY.value,
                    decision.rounded_qty,
                    "MARKET",
                    explanation=intent.rationale,
                )
                if order_id:
                    self.pending_microstructure[order_id] = self._pending_micro_record(instrument, decision)
            return self._execution_record(decision)
        if delta_value < 0 or intent.action in {"SELL", "EXIT", "REDUCE"}:
            decision = self.microstructure_guard.prepare_order(
                instrument=instrument,
                side="SELL",
                raw_qty=quantity,
                held_qty=self.long_qty[instrument],
                trade_date=self.current_time,
                candle=data,
                full_exit=intent.action == "EXIT" or intent.target_weight == 0,
            )
            if decision.executable:
                order_id = await self.place_order(
                    instrument,
                    Side.SELL.value,
                    decision.rounded_qty,
                    "MARKET",
                    explanation=intent.rationale,
                )
                if order_id:
                    self.pending_microstructure[order_id] = self._pending_micro_record(instrument, decision)
            return self._execution_record(decision)
        return base_record

    async def on_trade_execution(self, trade_data: dict[str, Any]) -> None:
        await super().on_trade_execution(trade_data)
        order_id = trade_data.get("order_id")
        micro = self.pending_microstructure.pop(order_id, None) if order_id else None
        if not micro:
            return
        side = "BUY" if trade_data.get("role") == "BUYER" else "SELL"
        qty = int(trade_data.get("quantity") or micro.get("rounded_qty") or 0)
        instrument = trade_data.get("instrument") or micro.get("instrument")
        trade_date = self.current_time or micro.get("trade_date")
        if instrument and qty > 0:
            self.microstructure_guard.record_filled(instrument, side, qty, trade_date)
        fee = float(micro.get("total_transaction_cost") or 0.0)
        if fee > 0:
            self.cash -= fee
            self.microstructure_guard.stats["accounting_transaction_cost"] = (
                self.microstructure_guard.stats.get("accounting_transaction_cost", 0.0) + fee
            )
            if self.metrics.trade_history:
                self.metrics.trade_history[-1]["transaction_cost"] = fee

    def _apply_position_cap(
        self, intent: TradingIntent, regime_state: RegimeState | None
    ) -> TradingIntent:
        if not regime_state or intent.target_weight <= regime_state.max_position_weight:
            return intent
        return TradingIntent(**{
            **intent.as_dict(),
            "target_weight": regime_state.max_position_weight,
            "rationale": f"{intent.rationale}\n[MACRO_CAP] max_position_weight={regime_state.max_position_weight:.4f}",
        })

    def _no_risk_record(self, regime_state: RegimeState | None) -> dict[str, Any]:
        return {
            "risk_engine": "disabled",
            "risk_metric": self.risk_metric,
            "veto": False,
            "veto_mask": False,
            "decay_applied": False,
            "risk_rewrite_reason": None,
            "veto_reason": None,
            "cvar_95_60d": None,
            "var_95_60d": None,
            "expected_mdd_60d": None,
            "risk_sample_size": 0,
            "max_position_weight": regime_state.max_position_weight if regime_state else None,
            "cvar_budget": regime_state.cvar_budget if regime_state else None,
        }

    def _pending_micro_record(self, instrument: str, decision) -> dict[str, Any]:
        return {
            **self._execution_record(decision),
            "instrument": instrument,
            "trade_date": self.current_time.isoformat() if self.current_time else None,
            "total_transaction_cost": decision.transaction_cost.total,
        }

    def _record_bridge_decision(
        self,
        instrument: str,
        compact_state: dict[str, Any],
        intent: TradingIntent,
        risk_record: dict[str, Any],
        activation_record: dict[str, Any] | None = None,
        execution_record: dict[str, Any] | None = None,
        raw_intent: TradingIntent | None = None,
    ) -> None:
        self.memory_pool.append({
            "timestamp": compact_state.get("timestamp"),
            "instrument": instrument,
            "action": intent.action,
            "target_weight": intent.target_weight,
            "confidence": intent.confidence,
            "risk": risk_record,
        })
        self.utility_memory.add(MemoryItem(
            event_time=compact_state.get("timestamp") or "",
            ticker=instrument,
            regime=compact_state.get("regime") or "unknown",
            action=intent.action,
            target_weight=intent.target_weight,
            max_drawdown_after_decision=compact_state.get("drawdown_from_60d_high"),
            veto_triggered=bool(risk_record.get("veto_mask") or risk_record.get("veto")),
            was_black_swan=abs(float(compact_state.get("return_1d") or 0.0)) > 0.08,
            text_summary=f"{intent.action} target={intent.target_weight:.4f} risk={risk_record.get('risk_rewrite_reason')}",
            metadata={"confidence": intent.confidence},
        ))
        memory_trace = compact_state.get("memory", {}) if isinstance(compact_state.get("memory"), dict) else {}
        regime_state = self.regime_cache.get(instrument) if self.macro_enabled else None
        record = {
            "date": self.current_time.strftime("%Y-%m-%d") if self.current_time else None,
            "ticker": instrument,
            "config_hash": self.config_hash,
            "macro_enabled": self.macro_enabled,
            "macro_disabled": not self.macro_enabled,
            "dynamic_memory_enabled": self.dynamic_memory_enabled,
            "risk_veto_enabled": self.risk_veto_enabled,
            "risk_metric": self.risk_metric,
            "microstructure_enabled": self.microstructure_guard.config.enabled,
            "regime": compact_state.get("regime"),
            "max_position_weight": regime_state.max_position_weight if regime_state else risk_record.get("max_position_weight"),
            "cvar_budget": regime_state.cvar_budget if regime_state else risk_record.get("cvar_budget"),
            "raw_intent": raw_intent.as_dict() if raw_intent else None,
            "risk_adjusted_intent": intent.as_dict(),
            "veto_mask": bool(risk_record.get("veto_mask") or risk_record.get("veto")),
            "decay_applied": bool(risk_record.get("decay_applied")),
            "risk_rewrite_reason": risk_record.get("risk_rewrite_reason"),
            "cvar_95_60d": risk_record.get("cvar_95_60d"),
            "var_95_60d": risk_record.get("var_95_60d"),
            "expected_mdd_60d": risk_record.get("expected_mdd_60d"),
            "risk_sample_size": risk_record.get("risk_sample_size", 0),
            "memory_items_used": memory_trace.get("memory_items_used", 0),
            "memory_pruned_count": memory_trace.get("memory_pruned_count", 0),
            "memory_scores": memory_trace.get("memory_scores", {}),
            "crisis_pinned_count": memory_trace.get("crisis_pinned_count", 0),
            **(activation_record or {}),
            **risk_record,
            **(execution_record or {}),
            "microstructure_stats": self.microstructure_guard.stats,
            "compact_state": compact_state,
        }
        self.trace_exporter.write(record)

    def _execution_record(self, decision) -> dict[str, Any]:
        return {
            "order_executable": decision.executable,
            "microstructure_block_reason": decision.block_reason,
            "limit_hit_state": decision.limit_hit_state,
            "raw_qty": decision.raw_qty,
            "rounded_qty": decision.rounded_qty,
            "used_price_lane": decision.used_price_lane,
            "commission": decision.transaction_cost.commission,
            "stamp_duty": decision.transaction_cost.stamp_duty,
            "slippage": decision.transaction_cost.slippage,
            "total_transaction_cost": decision.transaction_cost.total,
        }

    async def _publish_decision_done(self) -> None:
        if MessageType is None:
            return
        await self.publish_time(
            msg_type=MessageType.DECISION_RESPONSE,
            payload={"tick_id": self.current_tick_id},
            routing_key="simulation_clock",
        )
