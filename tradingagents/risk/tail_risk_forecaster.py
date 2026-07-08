"""Training-free historical tail-risk forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tradingagents.trading_intent import TradingIntent


@dataclass
class TailRiskEstimate:
    cvar_95_60d: float
    var_95_60d: float
    expected_mdd_60d: float
    sample_size: int
    method: str
    confidence: float


@dataclass
class TailRiskDecision:
    intent: TradingIntent
    estimate: TailRiskEstimate
    veto_mask: bool
    veto_reason: str | None
    adjusted_target_weight: float
    low_sample_fallback: bool = False


class TailRiskForecaster:
    def __init__(
        self,
        alpha: float = 0.95,
        horizon_days: int = 60,
        risk_budget: float = 0.12,
        max_mdd: float = 0.2,
        min_sample_size: int = 80,
        max_single_name_weight: float = 0.35,
    ) -> None:
        self.alpha = alpha
        self.horizon_days = horizon_days
        self.risk_budget = risk_budget
        self.max_mdd = max_mdd
        self.min_sample_size = min_sample_size
        self.max_single_name_weight = max_single_name_weight

    def estimate(self, returns: list[float], candidate_weight: float) -> TailRiskEstimate:
        clean = np.array([float(r) for r in returns if r is not None and np.isfinite(float(r))], dtype=float)
        if clean.size < self.min_sample_size:
            vol = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
            scaled = -abs(candidate_weight) * vol * np.sqrt(self.horizon_days)
            return TailRiskEstimate(
                cvar_95_60d=scaled * 1.25,
                var_95_60d=scaled,
                expected_mdd_60d=abs(scaled) * 1.5,
                sample_size=int(clean.size),
                method="vol_scaled_low_sample_fallback",
                confidence=max(0.1, min(0.5, clean.size / max(1, self.min_sample_size))),
            )
        weighted = clean * abs(candidate_weight)
        var = float(np.quantile(weighted, 1 - self.alpha))
        tail = weighted[weighted <= var]
        cvar = float(np.mean(tail)) if tail.size else var
        expected_mdd = self._expected_mdd(weighted)
        return TailRiskEstimate(
            cvar_95_60d=round(cvar, 6),
            var_95_60d=round(var, 6),
            expected_mdd_60d=round(expected_mdd, 6),
            sample_size=int(clean.size),
            method="historical_cvar",
            confidence=min(1.0, clean.size / (self.min_sample_size * 3)),
        )

    def veto_or_decay(
        self,
        intent: TradingIntent,
        returns: list[float],
        risk_budget: float | None = None,
    ) -> TailRiskDecision:
        budget = self.risk_budget if risk_budget is None else risk_budget
        candidate = min(intent.target_weight, self.max_single_name_weight)
        estimate = self.estimate(returns, candidate)
        low_sample = estimate.method.endswith("fallback")
        adjusted = candidate
        reason = None
        veto = False
        if estimate.cvar_95_60d < -abs(budget):
            veto = True
            reason = "CVaR_95_60d exceeds risk budget"
            adjusted = min(candidate, max(0.0, budget / max(abs(estimate.cvar_95_60d), 1e-9) * candidate))
        if estimate.expected_mdd_60d > self.max_mdd:
            reason = "expected_mdd_60d exceeds threshold"
            adjusted = min(adjusted, candidate * 0.5)
        adjusted_intent = TradingIntent(**{**intent.as_dict(), "target_weight": adjusted})
        return TailRiskDecision(
            intent=adjusted_intent,
            estimate=estimate,
            veto_mask=veto,
            veto_reason=reason,
            adjusted_target_weight=adjusted,
            low_sample_fallback=low_sample,
        )

    def _expected_mdd(self, weighted_returns: np.ndarray) -> float:
        if weighted_returns.size == 0:
            return 0.0
        window = min(self.horizon_days, weighted_returns.size)
        worst = 0.0
        for start in range(0, weighted_returns.size - window + 1):
            curve = np.cumprod(1 + weighted_returns[start : start + window])
            peak = np.maximum.accumulate(curve)
            drawdowns = curve / peak - 1
            worst = min(worst, float(np.min(drawdowns)))
        return abs(worst)
