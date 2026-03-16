"""
tests/conftest.py — 共有フィクスチャ
"""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.entities.order import Execution, OrderSide, SettleType
from src.domain.entities.position import Position
from src.domain.ports.exchange_port import ExchangePort
from src.domain.ports.signal_generator_port import SignalGeneratorPort
from src.domain.ports.state_repository_port import StateRepositoryPort


# ------------------------------------------------------------------ #
# OHLCV fixture                                                       #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """15分足 OHLCV DataFrame (100行)。ATR 計算に十分なデータ量。"""
    rng = np.random.default_rng(42)
    n = 100
    close = 5_000_000.0 + np.cumsum(rng.normal(0, 10_000, n))
    high = close + rng.uniform(5_000, 20_000, n)
    low = close - rng.uniform(5_000, 20_000, n)
    open_ = close + rng.normal(0, 5_000, n)
    volume = rng.uniform(0.1, 5.0, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"op": open_, "hi": high, "lo": low, "cl": close, "volume": volume},
        index=idx,
    )


# ------------------------------------------------------------------ #
# Mock ports                                                          #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_exchange() -> MagicMock:
    exchange = MagicMock(spec=ExchangePort)
    exchange.place_limit_order.return_value = 1001
    exchange.place_market_order.return_value = 1002
    exchange.place_limit_close_order.return_value = 1003
    exchange.cancel_orders.return_value = None
    exchange.get_recent_executions.return_value = []
    return exchange


@pytest.fixture
def mock_signal_buy() -> MagicMock:
    sig = MagicMock(spec=SignalGeneratorPort)
    sig.predict.return_value = 1.0  # always buy
    return sig


@pytest.fixture
def mock_signal_sell() -> MagicMock:
    sig = MagicMock(spec=SignalGeneratorPort)
    sig.predict.return_value = 0.0  # never sell
    return sig


@pytest.fixture
def mock_state_repo() -> MagicMock:
    repo = MagicMock(spec=StateRepositoryPort)
    repo.load_order_state.return_value = {}
    repo.load_position_state.return_value = {}
    return repo


# ------------------------------------------------------------------ #
# BotAgent fixture                                                    #
# ------------------------------------------------------------------ #

@pytest.fixture
def bot_agent(mock_exchange, mock_signal_buy, mock_signal_sell, mock_state_repo):
    from src.application.services.bot_agent import BotAgent
    return BotAgent(
        bot_id="0p186",
        atr_coeff=0.186,
        lot=0.01,
        atr_period=14,
        symbol="FX_BTC_JPY",
        pips=1,
        exchange=mock_exchange,
        signal_gen_buy=mock_signal_buy,
        signal_gen_sell=mock_signal_sell,
        state_repo=mock_state_repo,
        feature_names=[],
    )
