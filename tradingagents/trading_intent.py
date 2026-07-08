"""Structured trading-intent parsing without LLM-runtime imports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


Action = Literal["BUY", "SELL", "HOLD", "REDUCE", "EXIT"]


@dataclass
class TradingIntent:
    action: Action
    target_weight: float
    max_trade_weight: float | None
    confidence: float
    rationale: str
    rationale_type: str | None
    valid_until: str | None
    raw_decision: str
    parse_error: str | None = None
    instrument: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_weight": self.target_weight,
            "max_trade_weight": self.max_trade_weight,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "rationale_type": self.rationale_type,
            "valid_until": self.valid_until,
            "raw_decision": self.raw_decision,
            "parse_error": self.parse_error,
            "instrument": self.instrument,
        }


class TradingIntentParser:
    def parse(self, raw_decision: Any, current_weight: float = 0.0, instrument: str | None = None) -> TradingIntent:
        text = "" if raw_decision is None else str(raw_decision)
        try:
            payload = self._extract_json(text)
            if payload is not None:
                return self._from_payload(payload, text, current_weight, instrument)
            return self._from_text(text, current_weight, instrument)
        except Exception as exc:
            return TradingIntent(
                action="HOLD",
                target_weight=_clamp_weight(current_weight),
                max_trade_weight=None,
                confidence=0.0,
                rationale="parse failed; conservative HOLD",
                rationale_type="parse_error",
                valid_until=None,
                raw_decision=text,
                parse_error=str(exc),
                instrument=instrument,
            )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
        candidates = [fence.group(1)] if fence else []
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)
        elif stripped.startswith("{"):
            raise ValueError("malformed JSON trading intent")
        for candidate in candidates:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        return None

    def _from_payload(
        self,
        payload: dict[str, Any],
        raw: str,
        current_weight: float,
        instrument: str | None,
    ) -> TradingIntent:
        action = _normalize_action(payload.get("action"))
        target_weight = _normalize_target(action, payload.get("target_weight"), current_weight)
        return TradingIntent(
            action=action,
            target_weight=target_weight,
            max_trade_weight=_optional_float(payload.get("max_trade_weight")),
            confidence=_clamp_confidence(payload.get("confidence", 0.5)),
            rationale=str(payload.get("rationale") or "")[:4000],
            rationale_type=payload.get("rationale_type"),
            valid_until=payload.get("valid_until"),
            raw_decision=raw,
            instrument=instrument,
        )

    def _from_text(self, text: str, current_weight: float, instrument: str | None) -> TradingIntent:
        upper = text.upper()
        if re.search(r"\b(EXIT|LIQUIDATE|CLEAR|清仓)\b", upper):
            action: Action = "EXIT"
        elif re.search(r"\b(REDUCE|TRIM|减仓)\b", upper):
            action = "REDUCE"
        elif re.search(r"\bSELL\b|卖出", upper):
            action = "SELL"
        elif re.search(r"\bBUY\b|买入", upper):
            action = "BUY"
        else:
            action = "HOLD"
        return TradingIntent(
            action=action,
            target_weight=_normalize_target(action, _extract_weight(text), current_weight),
            max_trade_weight=None,
            confidence=_extract_confidence(text),
            rationale=text[:4000],
            rationale_type="natural_language",
            valid_until=None,
            raw_decision=text,
            instrument=instrument,
        )


def _normalize_action(value: Any) -> Action:
    upper = str(value or "HOLD").upper().strip()
    mapping = {"STRONG_BUY": "BUY", "STRONG SELL": "SELL", "STRONG_SELL": "SELL"}
    upper = mapping.get(upper, upper)
    if upper not in {"BUY", "SELL", "HOLD", "REDUCE", "EXIT"}:
        raise ValueError(f"unsupported action: {value!r}")
    return upper  # type: ignore[return-value]


def _normalize_target(action: Action, raw_weight: Any, current_weight: float) -> float:
    if action in {"SELL", "EXIT"} and raw_weight is None:
        return 0.0
    if action == "HOLD" and raw_weight is None:
        return _clamp_weight(current_weight)
    value = _optional_float(raw_weight)
    if value is None:
        if action == "BUY":
            return min(1.0, max(current_weight, 0.25))
        if action == "REDUCE":
            return max(0.0, current_weight * 0.5)
        return _clamp_weight(current_weight)
    if value > 1:
        value /= 100.0
    if action == "REDUCE" and value >= current_weight:
        value = max(0.0, current_weight * 0.5)
    return _clamp_weight(value)


def _extract_weight(text: str) -> float | None:
    patterns = [
        r"target[_\s-]*weight\D+([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s+)?portfolio",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value / 100.0 if value > 1 else value
    return None


def _extract_confidence(text: str) -> float:
    match = re.search(r"confidence\D+([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return _clamp_confidence(match.group(1) if match else 0.5)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _clamp_weight(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _clamp_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    if parsed > 1:
        parsed /= 100.0
    return max(0.0, min(parsed, 1.0))
