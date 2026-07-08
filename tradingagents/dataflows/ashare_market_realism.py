"""China A-share market-realism helpers.

These helpers are deliberately deterministic and independent from StockSim so
they can be used in unit tests, bridge adapters, and future exchange wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class MicrostructureConfig:
    enabled: bool = True
    use_raw_price_for_execution: bool = True
    use_adjusted_price_for_features: bool = True
    enforce_t_plus_one: bool = True
    enforce_price_limit: bool = True
    enforce_board_lot: bool = True
    tick_size: float = 0.01
    commission_bps: float = 3.0
    stamp_duty_bps_sell: float = 5.0
    slippage_model: str = "fixed_bps"
    slippage_bps: float = 5.0
    limit_hit_fill_model: str = "no_fill_on_one_word_limit"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MicrostructureConfig":
        if not raw:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass
class TransactionCostBreakdown:
    commission: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.stamp_duty + self.slippage


@dataclass
class MicrostructureDecision:
    executable: bool
    side: Side
    raw_qty: int
    rounded_qty: int
    execution_price: float
    used_price_lane: str
    block_reason: str | None = None
    limit_hit_state: str | None = None
    transaction_cost: TransactionCostBreakdown = field(default_factory=TransactionCostBreakdown)


@dataclass
class InventoryState:
    position_qty: int = 0
    sellable_qty: int = 0
    today_buy_qty: int = 0
    frozen_qty: int = 0
    trade_date: str | None = None


class SellableInventoryLedger:
    """T+1 sellable inventory ledger for China A-shares."""

    def __init__(self) -> None:
        self._states: dict[str, InventoryState] = {}

    def get(self, instrument: str) -> InventoryState:
        return self._states.setdefault(instrument, InventoryState())

    def sync_position(self, instrument: str, position_qty: int, trade_date: str | date | datetime) -> None:
        state = self.get(instrument)
        current_date = _date_key(trade_date)
        if state.trade_date != current_date:
            state.sellable_qty = min(position_qty, state.sellable_qty + state.frozen_qty)
            state.frozen_qty = 0
            state.today_buy_qty = 0
            state.trade_date = current_date
        state.position_qty = position_qty
        state.sellable_qty = min(state.sellable_qty, position_qty)

    def max_sellable(self, instrument: str, position_qty: int, trade_date: str | date | datetime) -> int:
        self.sync_position(instrument, position_qty, trade_date)
        return max(0, self.get(instrument).sellable_qty)

    def record_buy(self, instrument: str, qty: int, trade_date: str | date | datetime) -> None:
        state = self.get(instrument)
        self.sync_position(instrument, state.position_qty, trade_date)
        state.position_qty += qty
        state.today_buy_qty += qty
        state.frozen_qty += qty

    def record_sell(self, instrument: str, qty: int, trade_date: str | date | datetime) -> None:
        state = self.get(instrument)
        self.sync_position(instrument, state.position_qty, trade_date)
        sell_qty = min(qty, state.sellable_qty)
        state.position_qty = max(0, state.position_qty - sell_qty)
        state.sellable_qty = max(0, state.sellable_qty - sell_qty)


class BoardLotSizer:
    """Apply 100-share board-lot rules with odd-lot full-exit support."""

    board_lot: int = 100

    def size(self, side: Side, raw_qty: int, held_qty: int = 0, full_exit: bool = False) -> int:
        if raw_qty <= 0:
            return 0
        if side == "BUY":
            return (raw_qty // self.board_lot) * self.board_lot
        rounded = (raw_qty // self.board_lot) * self.board_lot
        residual = held_qty - rounded
        if full_exit and 0 < residual < self.board_lot and raw_qty >= held_qty:
            return held_qty
        return rounded


class ChinaPriceLimitGuard:
    """Detect one-word limit-up/limit-down states for A-share execution."""

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size

    def board_limit_pct(self, symbol: str, board: str | None = None) -> float:
        board_norm = (board or "").lower()
        code = "".join(ch for ch in str(symbol) if ch.isdigit())
        if "star" in board_norm or "科创" in board_norm or code.startswith("688"):
            return 0.20
        if "chinext" in board_norm or "创业" in board_norm or code.startswith("300"):
            return 0.20
        # TODO: ST/*ST, Beijing Exchange, IPO first days, and special-risk boards.
        return 0.10

    def limit_prices(self, prev_close: float, symbol: str, board: str | None = None) -> tuple[float, float]:
        pct = self.board_limit_pct(symbol, board)
        up = _round_tick(prev_close * (1 + pct), self.tick_size)
        down = _round_tick(prev_close * (1 - pct), self.tick_size)
        return up, down

    def evaluate(
        self,
        *,
        side: Side,
        symbol: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        prev_close: float | None,
        board: str | None = None,
    ) -> tuple[bool, str | None]:
        if not prev_close or prev_close <= 0:
            return True, None
        limit_up, limit_down = self.limit_prices(prev_close, symbol, board)
        one_word_up = all(_near(p, limit_up, self.tick_size) for p in (open_price, high_price, low_price, close_price))
        one_word_down = all(
            _near(p, limit_down, self.tick_size) for p in (open_price, high_price, low_price, close_price)
        )
        if one_word_up:
            return side != "BUY", "ONE_WORD_LIMIT_UP"
        if one_word_down:
            return side != "SELL", "ONE_WORD_LIMIT_DOWN"
        if high_price >= limit_up - self.tick_size / 2:
            return True, "LIMIT_UP"
        if low_price <= limit_down + self.tick_size / 2:
            return True, "LIMIT_DOWN"
        return True, None


class ChinaMicrostructureGuard:
    """Compose T+1, price-limit, board-lot, and cost rules."""

    def __init__(
        self,
        config: MicrostructureConfig | None = None,
        ledger: SellableInventoryLedger | None = None,
    ) -> None:
        self.config = config or MicrostructureConfig()
        self.ledger = ledger or SellableInventoryLedger()
        self.lot_sizer = BoardLotSizer()
        self.limit_guard = ChinaPriceLimitGuard(tick_size=self.config.tick_size)
        self.stats = {
            "t_plus_one_blocked_orders": 0,
            "limit_hit_blocked_orders": 0,
            "board_lot_rounded_orders": 0,
            "total_transaction_cost": 0.0,
        }

    def prepare_order(
        self,
        *,
        instrument: str,
        side: Side,
        raw_qty: int,
        held_qty: int,
        trade_date: str | date | datetime,
        candle: dict[str, Any],
        full_exit: bool = False,
    ) -> MicrostructureDecision:
        execution_price = float(candle.get("raw_execution_close") or candle.get("close") or 0.0)
        decision = MicrostructureDecision(
            executable=False,
            side=side,
            raw_qty=int(raw_qty),
            rounded_qty=0,
            execution_price=execution_price,
            used_price_lane="raw_execution_price",
        )
        if not self.config.enabled:
            decision.rounded_qty = max(0, int(raw_qty))
            decision.executable = decision.rounded_qty > 0 and execution_price > 0
            decision.used_price_lane = "default_price"
            return decision
        if execution_price <= 0:
            decision.block_reason = "NO_EXECUTION_PRICE"
            return decision

        if self.config.enforce_board_lot:
            rounded = self.lot_sizer.size(side, int(raw_qty), held_qty=held_qty, full_exit=full_exit)
        else:
            rounded = max(0, int(raw_qty))
        decision.rounded_qty = rounded
        if self.config.enforce_board_lot and rounded != int(raw_qty):
            self.stats["board_lot_rounded_orders"] += 1
        if rounded <= 0:
            decision.block_reason = "BOARD_LOT_ZERO"
            return decision

        if self.config.enforce_t_plus_one and side == "SELL":
            sellable = self.ledger.max_sellable(instrument, held_qty, trade_date)
            if sellable <= 0:
                self.stats["t_plus_one_blocked_orders"] += 1
                decision.block_reason = "T_PLUS_ONE"
                return decision
            if rounded > sellable:
                rounded = self.lot_sizer.size("SELL", sellable, held_qty=sellable, full_exit=full_exit)
                decision.rounded_qty = rounded
                if rounded <= 0:
                    self.stats["t_plus_one_blocked_orders"] += 1
                    decision.block_reason = "T_PLUS_ONE"
                    return decision

        if self.config.enforce_price_limit:
            can_fill, limit_state = self.limit_guard.evaluate(
                side=side,
                symbol=instrument,
                open_price=float(candle.get("raw_execution_open") or candle.get("open") or execution_price),
                high_price=float(candle.get("raw_execution_high") or candle.get("high") or execution_price),
                low_price=float(candle.get("raw_execution_low") or candle.get("low") or execution_price),
                close_price=execution_price,
                prev_close=_maybe_float(candle.get("raw_execution_prev_close") or candle.get("prev_close")),
                board=candle.get("board"),
            )
            decision.limit_hit_state = limit_state
            if not can_fill:
                self.stats["limit_hit_blocked_orders"] += 1
                decision.block_reason = limit_state
                return decision

        decision.transaction_cost = self.calculate_cost(side, rounded, execution_price)
        self.stats["total_transaction_cost"] += decision.transaction_cost.total
        decision.executable = True
        return decision

    def record_filled(self, instrument: str, side: Side, qty: int, trade_date: str | date | datetime) -> None:
        if not self.config.enabled:
            return
        if side == "BUY":
            self.ledger.record_buy(instrument, qty, trade_date)
        else:
            self.ledger.record_sell(instrument, qty, trade_date)

    def calculate_cost(self, side: Side, qty: int, price: float) -> TransactionCostBreakdown:
        notional = max(0.0, qty * price)
        commission = notional * self.config.commission_bps / 10000.0
        stamp = notional * self.config.stamp_duty_bps_sell / 10000.0 if side == "SELL" else 0.0
        slippage = notional * self.config.slippage_bps / 10000.0 if self.config.slippage_model == "fixed_bps" else 0.0
        return TransactionCostBreakdown(commission=commission, stamp_duty=stamp, slippage=slippage)


def _date_key(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _round_tick(value: float, tick_size: float) -> float:
    q = Decimal(str(tick_size))
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def _near(a: float, b: float, tick_size: float) -> bool:
    return abs(a - b) <= max(1e-9, tick_size / 2)


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
