"""A-share data adapter backed by AkShare."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Annotated, Any

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .symbol_utils import NoMarketDataError
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "akshare is required for A-share data. Install with `pip install akshare`."
        ) from exc
    return ak


class AShareDataAdapter:
    """AkShare-backed OHLCV adapter shared by TradingAgents and StockSim."""

    FIELD_MAP = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    PERIOD_MAP = {
        "1d": "daily",
        "1w": "weekly",
        "1mo": "monthly",
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
    }

    def __init__(
        self,
        base_cache_dir: str | None = None,
        max_retries: int = 3,
        base_delay: float = 1.5,
        timeout: int = 20,
    ) -> None:
        config = get_config()
        root = base_cache_dir or os.path.join(config["data_cache_dir"], "akshare")
        self.base_cache_dir = os.path.abspath(root)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.cache_dirs = {
            "ohlcv": os.path.join(self.base_cache_dir, "ohlcv"),
            "calendar": os.path.join(self.base_cache_dir, "calendar"),
            "news": os.path.join(self.base_cache_dir, "news"),
        }
        for folder in self.cache_dirs.values():
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        raw = str(symbol).upper().strip()
        raw = raw.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        raw = raw.removeprefix("SH").removeprefix("SZ").removeprefix("BJ")
        if not raw.isdigit() or len(raw) != 6:
            raise ValueError(f"Unsupported A-share symbol format: {symbol!r}")
        return raw

    @staticmethod
    def to_exchange_symbol(symbol: str) -> str:
        code = AShareDataAdapter.normalize_symbol(symbol)
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        return code

    @staticmethod
    def _date_key(date_like: str | datetime | None) -> str:
        if date_like is None:
            return datetime.now().strftime("%Y%m%d")
        return pd.to_datetime(date_like).strftime("%Y%m%d")

    def _cache_path(self, kind: str, filename: str) -> str:
        return os.path.join(self.cache_dirs[kind], filename)

    def _with_retry(self, label: str, fn):
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    "AkShare %s failed (%s); retrying in %.1fs [%s/%s]",
                    label,
                    exc,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)

    def get_trade_calendar(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DatetimeIndex:
        start_key = self._date_key(start_date or "1990-01-01")
        end_key = self._date_key(end_date or datetime.now())
        cache_path = self._cache_path("calendar", f"trade_calendar_{start_key}_{end_key}.csv")

        if use_cache and os.path.exists(cache_path):
            cal = pd.read_csv(cache_path)
        else:
            ak = _akshare()
            # CHANGED FOR DOMESTIC DATA: use real A-share trading calendar.
            cal = self._with_retry("trade_calendar", ak.tool_trade_date_hist_sina)
            cal.to_csv(cache_path, index=False, encoding="utf-8")

        date_col = "trade_date" if "trade_date" in cal.columns else cal.columns[0]
        dates = pd.to_datetime(cal[date_col], errors="coerce").dropna().dt.normalize()
        start_ts = pd.to_datetime(start_key).normalize()
        end_ts = pd.to_datetime(end_key).normalize()
        return pd.DatetimeIndex(dates[(dates >= start_ts) & (dates <= end_ts)]).sort_values()

    def normalize_akshare_ohlcv(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = raw.rename(columns=self.FIELD_MAP).copy()
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"AkShare OHLCV missing columns for {symbol}: {missing}")

        # CHANGED FOR DOMESTIC DATA: normalize Chinese columns and numeric types.
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return df[df["volume"] >= 0]

    def load_ohlcv_df(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        period: str = "daily",
        adjust: str = "qfq",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        code = self.normalize_symbol(symbol)
        start_key = self._date_key(start_date)
        end_key = self._date_key(end_date)
        period = self.PERIOD_MAP.get(period, period)
        cache_path = self._cache_path(
            "ohlcv",
            f"{safe_ticker_component(code)}_{period}_{adjust}_{start_key}_{end_key}.csv",
        )

        if use_cache and os.path.exists(cache_path):
            raw = pd.read_csv(cache_path)
        else:
            ak = _akshare()
            # CHANGED FOR DOMESTIC DATA: force qfq for long-horizon backtests.
            raw = self._with_retry(
                f"stock_zh_a_hist:{code}",
                lambda: ak.stock_zh_a_hist(
                    symbol=code,
                    period=period,
                    start_date=start_key,
                    end_date=end_key,
                    adjust=adjust,
                    timeout=self.timeout,
                ),
            )
            if raw is None or raw.empty:
                raise NoMarketDataError(symbol, code, f"AkShare returned no rows {start_key}-{end_key}")
            raw.to_csv(cache_path, index=False, encoding="utf-8")

        df = self.normalize_akshare_ohlcv(raw, symbol=symbol)
        # CHANGED FOR DOMESTIC DATA: align candles to China A-share trade calendar.
        calendar = self.get_trade_calendar(start_key, end_key)
        df = df[df["date"].dt.normalize().isin(calendar)]
        if df.empty:
            raise NoMarketDataError(symbol, code, "no rows after A-share trade calendar alignment")
        return df.sort_values("date").reset_index(drop=True)

    def load_aggregates(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str | None = None,
        end_date: str | None = None,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        period = self.PERIOD_MAP.get(interval.lower())
        if period is None:
            raise ValueError("AkShare adapter supports A-share daily/weekly/monthly bars only.")
        df = self.load_ohlcv_df(
            symbol,
            start_date or "1990-01-01",
            end_date or datetime.now().strftime("%Y-%m-%d"),
            period=period,
            adjust="qfq" if adjusted else "",
            use_cache=use_cache,
        )
        rows = [
            {
                "timestamp": row.date.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }
            for row in df.itertuples(index=False)
        ]
        if sort == "desc":
            rows.reverse()
        return rows[:limit]

    def get_stock_csv(self, symbol: str, start_date: str, end_date: str) -> str:
        df = self.load_ohlcv_df(symbol, start_date, end_date, period="daily", adjust="qfq")
        out = df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )[["Date", "Open", "High", "Low", "Close", "Volume"]]
        header = (
            f"# A-share qfq stock data for {self.to_exchange_symbol(symbol)} "
            f"from {start_date} to {end_date}\n"
            f"# Total records: {len(out)}\n"
            "# Source: AkShare stock_zh_a_hist(adjust='qfq') + Sina trade calendar\n\n"
        )
        return header + out.to_csv(index=False)

    def get_indicator_window(self, symbol: str, indicator: str, curr_date: str, look_back_days: int) -> str:
        start = (pd.to_datetime(curr_date) - relativedelta(days=max(look_back_days, 260))).strftime(
            "%Y-%m-%d"
        )
        df = self.load_ohlcv_df(symbol, start, curr_date, period="daily", adjust="qfq")
        stock_df = df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )[["Date", "Open", "High", "Low", "Close", "Volume"]]
        wrapped = wrap(stock_df)
        wrapped["Date"] = pd.to_datetime(wrapped["Date"]).dt.strftime("%Y-%m-%d")
        wrapped[indicator]
        cutoff = pd.to_datetime(curr_date) - relativedelta(days=look_back_days)
        recent = wrapped[pd.to_datetime(wrapped["Date"]) >= cutoff]
        lines = []
        for _, row in recent.iterrows():
            value = row.get(indicator, "N/A")
            lines.append(f"{row['Date']}: {value}")
        return (
            f"## {indicator} values for A-share {self.to_exchange_symbol(symbol)} "
            f"from {cutoff.strftime('%Y-%m-%d')} to {curr_date}\n\n"
            + "\n".join(lines)
        )

    def load_all_corporate_fundamentals(
        self, symbol: str, as_of_date: str | None = None, use_cache: bool = True
    ) -> dict[str, Any]:
        return {
            "source": "akshare",
            "symbol": self.to_exchange_symbol(symbol),
            "as_of_date": as_of_date,
            "note": "A-share fundamentals are optional; OHLCV uses qfq adjusted prices.",
        }

    def load_news(self, symbol: str, limit: int = 50, use_cache: bool = True, **_: Any) -> list[dict[str, Any]]:
        return []


def get_ashare_stock_data(
    symbol: Annotated[str, "A-share ticker, e.g. 600519.SH or 600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    return AShareDataAdapter().get_stock_csv(symbol, start_date, end_date)


def get_ashare_indicator(
    symbol: Annotated[str, "A-share ticker, e.g. 600519.SH or 600519"],
    indicator: Annotated[str, "stockstats indicator name"],
    curr_date: Annotated[str, "current date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "lookback calendar days"],
) -> str:
    return AShareDataAdapter().get_indicator_window(symbol, indicator, curr_date, look_back_days)


def get_ashare_fundamentals(
    symbol: Annotated[str, "A-share ticker"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    payload = AShareDataAdapter().load_all_corporate_fundamentals(symbol, curr_date)
    return json.dumps(payload, ensure_ascii=False, indent=2)
