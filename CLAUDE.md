# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (TA-Lib C library must be installed first)
pip install -e ".[dev]"

# Run the trading bot
python -m src.entrypoints.main
# or via pyproject.toml script:
bot

# Retrain all models (uses BitFlyer REST + cached data)
python -m src.entrypoints.retrain.retrain_main --trials 50

# Retrain with CSV bootstrap (required for >31 days of history)
python -m src.entrypoints.retrain.retrain_main --csv /work/data/BitFlyer_BTCJPY_1h.csv --trials 100

# Retrain a single bot
python -m src.entrypoints.retrain.retrain_main --bot-id 0p186 --trials 50

# Docker
docker compose up bot
docker compose --profile retrain run retrain

# Lint
ruff check src/
ruff format src/

# Type check
mypy src/

# Tests (no test suite exists yet; tests go in tests/)
pytest tests/
pytest tests/unit/test_bot_agent.py  # single file
```

## Architecture

The codebase follows **Hexagonal (Ports & Adapters) architecture** with four layers. The key rule: **inner layers never import from outer ones**.

```
domain/         ← no external imports; pure Python dataclasses + ABCs
application/    ← imports only from domain/
infrastructure/ ← implements domain ports; imports third-party libs
entrypoints/    ← only place that imports from all layers simultaneously
```

### Domain layer (`src/domain/`)

- `entities/` — frozen dataclasses: `Order`, `Execution`, `Position`, `PositionSummary`, `Candle`
- `ports/` — Abstract Base Classes (the "interfaces"):
  - `ExchangePort` — all exchange operations (place/cancel orders, get positions/balance/executions)
  - `SignalGeneratorPort` — `predict(df, feature_names) -> float` (positive = signal)
  - `StateRepositoryPort` — order/position persistence per bot_id
  - `OHLCVRepositoryPort` — load / save / fetch_and_update candles
  - `ModelRepositoryPort` — load/save joblib model pairs

### Application layer (`src/application/`)

- `BotAgent` — single bot; holds `ExchangePort`, `SignalGeneratorPort`, `StateRepositoryPort` via constructor injection. Key methods: `sync_state_from_executions()`, `compute_order_prices()`, `entry()`, `exit_limit()`, `exit_market()`, `run_cycle()`
- `BotOrchestrator` — manages N `BotAgent` instances; owns the main `while True` loop with 15-minute bar detection and Wednesday maintenance guard
- `LotAllocator` — distributes JPY balance across bots; always uses `[-2]` candle (penultimate confirmed bar)
- `MaintenanceScheduler` — encapsulates the Wednesday 14:00–17:00 JST logic

### Infrastructure layer (`src/infrastructure/`)

- `exchange/bitflyer/` — `BitFlyerExchangeAdapter` (implements `ExchangePort`), `BitFlyerDataProvider` (builds OHLCV from WebSocket + REST executions), `BitFlyerAuthenticator` (HMAC-SHA256, Unix seconds timestamp)
- `exchange/gmo/` — `GMOExchangeAdapter` + `GMORetryHandler` (ERR-5008/5009 logic, centralised from the original 7 copy-pasted blocks) + `GMOAuthenticator` (millisecond timestamp)
- `persistence/` — `PickleStateRepository`, `BitFlyerOHLCVRepository`, `JoblibModelRepository`
- `feature_engineering/TALibFeatureCalculator` — **must stay bit-for-bit identical to the original `src/richman_features.py`**; the trained `.xz` models depend on the exact feature names and computation
- `ml/LightGBMSignalGenerator` — wraps a loaded joblib model; always uses `iloc[-2]` (same convention as the original)

### Entrypoints (`src/entrypoints/`)

- `main.py` — **Composition Root**: the only file that wires all layers together. Instantiates concrete implementations and injects them into `BotOrchestrator`. Edit here to swap exchanges or repositories.
- `retrain/` — standalone retraining pipeline: `HistoricalDataFetcher` → `TALibFeatureCalculator` → `LabelGenerator` → `ModelTrainer` (Optuna + `TimeSeriesSplit`) → `ModelEvaluator` → `JoblibModelRepository.save_model_pair()`

## Critical conventions

**`[-2]` indexing** — both signal prediction and order price computation use `df.iloc[-2]`, not `-1`. This refers to the last *closed* bar, not the forming one. Do not change this.

**Feature list immutability** — `features/features_default.pkl` is a `List[str]` that must match the column names produced by `TALibFeatureCalculator.calculate()`. Changing any feature name or computation invalidates all existing `.xz` models without retraining.

**Model filename convention** — `buy_<bot_id>.xz` / `sell_<bot_id>.xz` where `bot_id` encodes the ATR coefficient with `p` as decimal point (e.g. `0p186` → `0.186`). `JoblibModelRepository.get_atr_coeff()` parses this.

**Exchange is BitFlyer only** — GMO Coin adapter exists for reference but is not wired into `main.py`. `EXCHANGE_TYPE=bitflyer` is the only supported runtime value.

**BitFlyer OHLCV limitation** — BitFlyer has no native candle API. `BitFlyerDataProvider` builds candles from `/v1/getexecutions` (REST, max 31 days) and `lightning_executions_*` (WebSocket, real-time). For retraining with longer history, import a CSV first via `--csv`.

## Configuration

All settings are read from `.env` via `src/config/settings.py` (pydantic-settings). Copy `.env.example` to `.env`. Key variables: `BITFLYER_API_KEY`, `BITFLYER_SECRET_KEY`, `SYMBOL` (`FX_BTC_JPY` recommended), `AVAILABLE_MARGIN` (0.0–0.85).

## Legacy files

`src/start_all_bots.py`, `src/gmocoin.py`, `src/richman_features.py`, `src/config.py` are the **original 2022 implementation** kept for reference. They are not imported by the new architecture.
