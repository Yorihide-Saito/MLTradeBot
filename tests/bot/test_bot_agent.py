"""
tests/unit/test_bot_agent.py — BotAgent の単体テスト
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.domain.entities.order import Execution, OrderSide, SettleType
from src.domain.entities.position import Position

_TS = datetime(2024, 1, 1)


class TestSyncStateFromExecutions:
    def test_open_execution_creates_position(self, bot_agent):
        exc = Execution(
            order_id=1001,
            position_id=2001,
            side=OrderSide.BUY,
            settle_type=SettleType.OPEN,
            size=0.01,
            price=5_000_000,
            loss_gain=0.0,
            timestamp=_TS,
        )
        bot_agent._order_state[1001] = ("BUY", 0.01, 5_000_000)
        bot_agent.sync_state_from_executions([exc])

        assert 2001 in bot_agent._position_state
        pos = bot_agent._position_state[2001]
        assert pos.side == OrderSide.BUY
        assert pos.entry_price == 5_000_000

    def test_close_execution_removes_position(self, bot_agent):
        bot_agent._position_state[2001] = Position(
            position_id=2001,
            settle_type=SettleType.OPEN,
            side=OrderSide.BUY,
            size=0.01,
            entry_price=5_000_000,
        )
        exc = Execution(
            order_id=9999,
            position_id=2001,
            side=OrderSide.SELL,
            settle_type=SettleType.CLOSE,
            size=0.01,
            price=5_100_000,
            loss_gain=0.0,
            timestamp=_TS,
        )
        bot_agent.sync_state_from_executions([exc])
        assert 2001 not in bot_agent._position_state

    def test_state_persisted_after_sync(self, bot_agent, mock_state_repo):
        bot_agent.sync_state_from_executions([])
        mock_state_repo.save_order_state.assert_called_once()
        mock_state_repo.save_position_state.assert_called_once()


class TestCancelPendingOrders:
    def test_cancels_all_order_ids(self, bot_agent, mock_exchange):
        bot_agent._order_state = {101: ("BUY", 0.01, 5_000_000), 102: ("SELL", 0.01, 5_100_000)}
        bot_agent.cancel_pending_orders()
        called_ids = mock_exchange.cancel_orders.call_args[0][0]
        assert set(called_ids) == {101, 102}
        assert bot_agent._order_state == {}

    def test_no_api_call_when_no_orders(self, bot_agent, mock_exchange):
        bot_agent._order_state = {}
        bot_agent.cancel_pending_orders()
        mock_exchange.cancel_orders.assert_not_called()


class TestComputeSignals:
    def test_buy_signal_when_buy_score_positive(self, bot_agent, mock_signal_buy, sample_ohlcv):
        mock_signal_buy.predict.return_value = 0.7
        signal = bot_agent.compute_signals(sample_ohlcv)
        assert signal.buy_signal is True
        assert signal.buy_score == pytest.approx(0.7)

    def test_no_buy_signal_when_score_zero(self, bot_agent, mock_signal_buy, sample_ohlcv):
        mock_signal_buy.predict.return_value = 0.0
        signal = bot_agent.compute_signals(sample_ohlcv)
        assert signal.buy_signal is False


class TestEntry:
    def test_places_buy_order_when_buy_signal(self, bot_agent, mock_exchange, mock_signal_buy, sample_ohlcv):
        mock_signal_buy.predict.return_value = 1.0
        bot_agent.entry(sample_ohlcv, buy_price=4_900_000, sell_price=5_100_000)
        mock_exchange.place_limit_order.assert_called_once_with(
            "FX_BTC_JPY", OrderSide.BUY, 0.01, 4_900_000
        )

    def test_no_entry_when_position_exists(self, bot_agent, mock_exchange):
        bot_agent._position_state[2001] = Position(
            position_id=2001,
            settle_type=SettleType.OPEN,
            side=OrderSide.BUY,
            size=0.01,
            entry_price=5_000_000,
        )
        bot_agent.entry(MagicMock(), buy_price=4_900_000, sell_price=5_100_000)
        mock_exchange.place_limit_order.assert_not_called()

    def test_no_entry_when_lot_too_large(self, bot_agent, mock_exchange, sample_ohlcv):
        bot_agent.lot = 0.5  # 0.1 以上
        bot_agent.entry(sample_ohlcv, buy_price=4_900_000, sell_price=5_100_000)
        mock_exchange.place_limit_order.assert_not_called()


class TestExitLimit:
    def test_places_close_order_for_buy_position(self, bot_agent, mock_exchange):
        bot_agent._position_state[2001] = Position(
            position_id=2001,
            settle_type=SettleType.OPEN,
            side=OrderSide.BUY,
            size=0.01,
            entry_price=5_000_000,
        )
        bot_agent.exit_limit(buy_price=4_900_000, sell_price=5_100_000)
        mock_exchange.place_limit_close_order.assert_called_once_with(
            "FX_BTC_JPY", OrderSide.SELL, 0.01, 5_100_000, 2001
        )

    def test_places_close_order_for_sell_position(self, bot_agent, mock_exchange):
        bot_agent._position_state[2001] = Position(
            position_id=2001,
            settle_type=SettleType.OPEN,
            side=OrderSide.SELL,
            size=0.01,
            entry_price=5_000_000,
        )
        bot_agent.exit_limit(buy_price=4_900_000, sell_price=5_100_000)
        mock_exchange.place_limit_close_order.assert_called_once_with(
            "FX_BTC_JPY", OrderSide.BUY, 0.01, 4_900_000, 2001
        )


class TestExitMarket:
    def test_market_close_called_for_each_position(self, bot_agent, mock_exchange):
        bot_agent._position_state = {
            2001: Position(2001, SettleType.OPEN, OrderSide.BUY, 0.01, 5_000_000),
            2002: Position(2002, SettleType.OPEN, OrderSide.SELL, 0.01, 5_100_000),
        }
        bot_agent.exit_market()
        assert mock_exchange.place_market_close_order.call_count == 2
