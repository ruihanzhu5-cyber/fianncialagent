# Long-Horizon A-share Financial Multi-Agent Benchmark

This repository is a training-free research codebase for long-horizon China A-share financial multi-agent backtesting. It bridges TradingAgents and StockSim into one paper-oriented experiment pipeline.

This repository keeps TradingAgents as the only LLM decision brain. StockSim is used only as the event-driven simulation, execution, accounting, and metrics substrate.

## What This Repo Is

- A reproducible A-share benchmark framework for TradingAgents-style financial multi-agent decisions.
- A StockSim-backed event loop for market data, order execution, portfolio accounting, and metrics.
- A training-free long-horizon control layer with macro regime caps, dynamic memory, tail-risk veto, A-share microstructure realism, and decision traces.

## What This Repo Is Not

- It is not a fine-tuning or trained-model project.
- It is not a pure upstream TradingAgents release.
- It is not a pure upstream StockSim release.
- Do not run upstream StockSim LLMTradingAgent for paper experiments. It has been removed/disabled to avoid duplicate LLM decision systems.
- `cli/` is retained only for upstream TradingAgents compatibility and manual debugging. It is not used in paper experiments.

## Core Runtime Path

```text
configs/stocksim_tradingagents_ashare_long_horizon.yaml
        -> scripts/run_ablation.py  (optional overlay merge)
        -> third_party/StockSim/main_launcher.py
        -> CandleBasedExchangeAgent
        -> TradingAgentsStockSimAgent in bridge_simulation.py
        -> TradingAgentsGraph.propagate()
        -> TradingIntentParser
        -> MacroRegimeAgent / UtilityMemoryManager / TailRiskForecaster
        -> ChinaMicrostructureGuard
        -> StockSim order execution and portfolio accounting
        -> metrics + traces
```

## Installation

Use the integrated dependency entry only:

```bash
pip install -r requirements-merged.txt
```

`requirements.txt` delegates to `requirements-merged.txt`. The upstream StockSim requirements are retained only as `third_party/StockSim/requirements-upstream.txt` for reference.

## A-share Experiment

```bash
python third_party/StockSim/main_launcher.py configs/stocksim_tradingagents_ashare_long_horizon.yaml
```

Set `OPENAI_API_KEY` and a reachable RabbitMQ host. AkShare A-share market data does not require Polygon or Alpha Vantage keys.

## Ablation Experiments

Run an ablation by merging the base config with an overlay:

```bash
python scripts/run_ablation.py --base configs/stocksim_tradingagents_ashare_long_horizon.yaml --overlay configs/ablations/ashare_A0_full.yaml
```

Use `--dry-run` to materialize the merged YAML without launching StockSim.

- `A0_full`: MacroRegime + DynamicMemory + TailRiskVeto + ChinaMicrostructure.
- `A1_no_macro`: disables macro regime control.
- `A2_no_dynamic_memory`: disables utility-scored memory retrieval.
- `A3_no_tail_risk_veto`: disables risk intent rewrites.
- `A4_no_long_horizon`: disables macro, dynamic memory, tail-risk veto, and bridge context injection.
- `A5_no_china_microstructure`: disables A-share microstructure guards.
- `A6_daily_full_debate`: forces daily full debate.
- `A7_rule_only_risk`: uses drawdown-threshold risk instead of historical CVaR.

## Output Files

Runtime outputs are intentionally ignored by Git:

- `logs/`
- `metrics/`
- `traces/`
- `cache/`
- temporary merged ablation YAML files

Decision traces are JSONL rows intended for ablation tables and auditability.

## Retained Upstream Components

- TradingAgents graph, agents, dataflows, LLM clients, and memory utilities needed by `TradingAgentsGraph`.
- StockSim exchange, simulation clock, accounting utilities, order utilities, and core `TraderAgent`.
- Traditional baseline traders: Buy-and-Hold, SMA, and MACD.

## Removed Upstream Components

- StockSim's separate LLMTradingAgent and analyst/wrapper stack.
- Redundant StockSim baseline traders not used by the paper path.
- Upstream-only changelog content.

The cleaned architecture is intentionally narrow: TradingAgents decides; StockSim simulates and accounts; the bridge adds long-horizon A-share controls without training new model weights.
