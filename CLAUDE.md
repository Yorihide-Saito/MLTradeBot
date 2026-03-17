# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (TA-Lib C library must be installed first)
pip install -e ".[dev]"

# Run the trading bot
python -m bot.src.main
# or via pyproject.toml script:
bot

# Retrain all models — Binance (推奨)
python -m training.src.main --binance --days 365 --trials 100

# Retrain with CSV bootstrap (BitFlyer 形式)
python -m training.src.main --csv /work/data/BitFlyer_BTCJPY_1h.csv --trials 100

# Retrain a single bot
python -m training.src.main --binance --days 365 --bot-id 0p186 --trials 50

# Docker
docker compose up bot
docker compose --profile retrain run retrain

# Lint
ruff check src/ bot/ training/ back_test/
ruff format src/ bot/ training/ back_test/

# Type check
mypy src/ bot/ training/ back_test/

# Tests (testpaths configured in pyproject.toml)
pytest
pytest bot/unit_test/test_bot_agent.py  # single file
pytest -x                               # stop on first failure
```

## Directory Structure

```
bot/
  src/main.py         ← Composition Root: wires all layers, starts BotOrchestrator
  unit_test/          ← bot & shared infrastructure tests

training/
  src/                ← retraining pipeline (independent of bot runtime)
    main.py           ← CLI: --binance / --csv / --days / --trials / --bot-id
    pipeline.py       ← RetrainPipeline orchestrator
    binance_data_fetcher.py  ← Binance BTCUSDT × USDJPY → BTCJPY
    data_fetcher.py   ← BitFlyer REST (max 31 days)
    label_generator.py
    model_trainer.py
    model_evaluator.py
  unit_test/          ← training-specific tests

back_test/
  src/                ← OHLCV backtesting (未実装)
  unit_test/

src/                  ← 共有レイヤー (domain / application / infrastructure / config)
model_buy/            ← buy_<bot_id>.xz (gitignore対象, .gitkeep で追跡)
model_sell/           ← sell_<bot_id>.xz (同上)
```

## Architecture

Hexagonal (Ports & Adapters). **Inner layers never import from outer ones.**

```
src/domain/         ← pure Python dataclasses + ABCs (no external deps)
src/application/    ← imports only from domain/
src/infrastructure/ ← implements domain ports; imports third-party libs
bot/src/ training/src/ back_test/src/  ← import from all layers
```

### Domain ports (`src/domain/ports/`)

- `ExchangePort` — place/cancel orders, get positions/balance/executions
- `SignalGeneratorPort` — `predict(df, feature_names) -> float` (positive = signal)
- `StateRepositoryPort` — order/position persistence per bot_id
- `OHLCVRepositoryPort` — load/save/fetch candles
- `ModelRepositoryPort` — load/save joblib model pairs

### Application services (`src/application/services/`)

- `BotAgent` — single bot; key methods: `sync_state_from_executions()`, `compute_order_prices()`, `entry()`, `exit_limit()`, `exit_market()`, `run_cycle()`
- `BotOrchestrator` — manages N `BotAgent` instances; main `while True` loop with 15-min bar detection and Wednesday maintenance guard
- `LotAllocator` — distributes JPY balance across bots; uses `[-2]` candle
- `MaintenanceScheduler` — Wednesday 14:00–17:00 JST logic

### Infrastructure (`src/infrastructure/`)

- `exchange/bitflyer/` — `BitFlyerExchangeAdapter`, `BitFlyerDataProvider` (WebSocket + REST), `BitFlyerAuthenticator` (HMAC-SHA256, Unix seconds)
- `exchange/gmo/` — GMO adapter (reference only, not wired)
- `persistence/` — `PickleStateRepository`, `BitFlyerOHLCVRepository`, `JoblibModelRepository`
- `feature_engineering/TALibFeatureCalculator` — **must stay bit-for-bit identical to `src/richman_features.py`**
- `ml/LightGBMSignalGenerator` — always uses `iloc[-2]`

## Critical conventions

**`[-2]` indexing** — signal prediction and order price computation use `df.iloc[-2]` (last *closed* bar). Do not change to `-1`.

**Feature list immutability** — `features/features_default.pkl` (`List[str]`) must match `TALibFeatureCalculator.calculate()` column names exactly. Changing any feature name/computation invalidates all existing `.xz` models.

**Model filename convention** — `buy_<bot_id>.xz` / `sell_<bot_id>.xz` where `bot_id` encodes ATR coefficient with `p` as decimal (e.g. `0p186` → `0.186`).

**Exchange is BitFlyer only** — `EXCHANGE_TYPE=bitflyer` is the only supported runtime value.

**BitFlyer OHLCV limitation** — No native candle API. Candles are built from `/v1/getexecutions` (max 31 days) + WebSocket. Use `--binance` for longer training history.

## Configuration

`.env` via `src/config/settings.py` (pydantic-settings). Key variables: `BITFLYER_API_KEY`, `BITFLYER_SECRET_KEY`, `SYMBOL` (`FX_BTC_JPY`), `AVAILABLE_MARGIN` (0.0–0.85).

## Legacy files

`src/start_all_bots.py`, `src/gmocoin.py`, `src/richman_features.py`, `src/config.py` — original 2022 implementation, kept for reference only.
