# Long-Horizon TradingAgents x StockSim

Training-free financial multi-agent research scaffold.

## Core Composition

- **Decision brain**: `TauricResearch/TradingAgents` provides analyst reports, bull/bear debate, risk debate, portfolio judgement, and `TradingAgentsGraph.propagate()`.
- **Evaluation substrate**: `harrypapadakis/StockSim` provides event-driven market simulation, order execution, portfolio accounting, and long-horizon metrics.
- **Bridge**: `bridge_simulation.py` exposes `TradingAgentsStockSimAgent`, a StockSim trader whose market-data callback calls TradingAgents and maps the final decision into StockSim orders.

## Long-Horizon Inserts

- `# CHANGED FOR LONG-HORIZON`: monthly/weekly **MacroRegimeAgent** cache before micro trading decisions.
- `# CHANGED FOR LONG-HORIZON`: pre-order **RiskVetoAgent** that can reduce or block high-risk target weights.
- Compact state passes OHLCV, indicators, portfolio value, drawdown, volatility, and memory ledger instead of raw full history.
- `# CHANGED FOR DOMESTIC DATA`: **AShareDataAdapter** routes China A-share OHLCV through AkShare with `qfq` adjustment and the Sina A-share trade calendar.

## Layout

```text
tradingagents/                         # upstream TradingAgents core
third_party/StockSim/                  # upstream StockSim core
bridge_simulation.py                   # non-invasive integration layer
configs/stocksim_tradingagents_long_horizon.yaml
configs/stocksim_tradingagents_ashare_long_horizon.yaml
requirements-merged.txt
```

## Run Skeleton

```bash
pip install -r requirements-merged.txt
python third_party/StockSim/main_launcher.py configs/stocksim_tradingagents_long_horizon.yaml
```

For China A-share experiments:

```bash
python third_party/StockSim/main_launcher.py configs/stocksim_tradingagents_ashare_long_horizon.yaml
```

Set `OPENAI_API_KEY` and a reachable RabbitMQ host. AkShare A-share runs do not require `POLYGON_API_KEY` or `ALPHA_VANTAGE_API_KEY`; legacy overseas configs still do.

## Metrics

StockSim exports metrics under `METRICS_OUTPUT_DIR` or `metrics/`, including:

- ROI
- Sharpe Ratio
- Annualized Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
