from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.application.dto.trading_signal_dto import TradingSignalDTO
from src.domain.entities.order import Execution, OrderSide, SettleType
from src.domain.entities.position import Position
from src.domain.ports.exchange_port import ExchangePort
from src.domain.ports.signal_generator_port import SignalGeneratorPort
from src.domain.ports.state_repository_port import StateRepositoryPort


class BotAgent:
    """単一ボットの取引ロジックを担うクラス。

    元コードの GMOBot を Clean Architecture に移植したもの。
    ExchangePort / SignalGeneratorPort / StateRepositoryPort を
    コンストラクタ注入することで取引所・モデルを差し替え可能にする。
    """

    def __init__(
        self,
        bot_id: str,
        atr_coeff: float,
        lot: float,
        atr_period: int,
        symbol: str,
        pips: int,
        exchange: ExchangePort,
        signal_gen_buy: SignalGeneratorPort,
        signal_gen_sell: SignalGeneratorPort,
        state_repo: StateRepositoryPort,
        feature_names: List[str],
    ) -> None:
        self.bot_id = bot_id
        self.atr_coeff = atr_coeff
        self.lot = lot
        self.atr_period = atr_period
        self.symbol = symbol
        self.pips = pips

        self._exchange = exchange
        self._signal_buy = signal_gen_buy
        self._signal_sell = signal_gen_sell
        self._state = state_repo
        self._feature_names = feature_names

        # 状態をキャッシュから復元
        self._order_state: Dict[int, tuple] = self._state.load_order_state(bot_id)
        self._position_state: Dict[int, Position] = self._state.load_position_state(bot_id)

    # ------------------------------------------------------------------ #
    # State synchronization                                               #
    # ------------------------------------------------------------------ #

    def sync_state_from_executions(self, executions: List[Execution]) -> None:
        """約定履歴を元に order / position の状態を更新する。

        元コードの GMOBot.update_open_positionIds() に相当。
        """
        executed_open_orders: set[int] = set()
        executed_close_positions: set[int] = set()

        for exc in executions:
            if exc.order_id in self._order_state and exc.settle_type == SettleType.OPEN:
                self._position_state[exc.position_id] = Position(
                    position_id=exc.position_id,
                    settle_type=SettleType.OPEN,
                    side=exc.side,
                    size=round(exc.size, 2),
                    entry_price=int(exc.price),
                )
                executed_open_orders.add(exc.order_id)

            if (
                exc.position_id in self._position_state
                and self._position_state[exc.position_id].settle_type == SettleType.OPEN
                and exc.settle_type == SettleType.CLOSE
            ):
                executed_close_positions.add(exc.position_id)

        for pos_id in executed_close_positions:
            self._position_state.pop(pos_id, None)

        self._persist_state()

    # ------------------------------------------------------------------ #
    # Order cancellation                                                  #
    # ------------------------------------------------------------------ #

    def cancel_pending_orders(self) -> None:
        """保留中の注文をすべてキャンセルする。

        元コードの GMOBot.cancel_orders() に相当。
        """
        order_ids = list(self._order_state.keys())
        if order_ids:
            self._exchange.cancel_orders(order_ids)
            self._order_state.clear()
        self._persist_state()

    # ------------------------------------------------------------------ #
    # Price calculation                                                   #
    # ------------------------------------------------------------------ #

    def compute_order_prices(self, df: pd.DataFrame) -> tuple[int, int]:
        """ATR に基づいて指値注文価格を計算する。

        元コードの GMOBot.get_latest_order_price() に相当。
        [-2] 行を使用する (直近確定足)。

        Returns:
            (buy_price, sell_price)
        """
        hi, lo, cl = df["hi"].values, df["lo"].values, df["cl"].values
        atr = talib.ATR(hi, lo, cl, timeperiod=self.atr_period)
        dist = atr * self.atr_coeff
        dist = np.maximum(1, (dist / self.pips).round().fillna(1)) * self.pips

        buy_price = int(cl[-2] - dist.iloc[-2])
        sell_price = int(cl[-2] + dist.iloc[-2])
        return buy_price, sell_price

    # ------------------------------------------------------------------ #
    # Signal generation                                                   #
    # ------------------------------------------------------------------ #

    def compute_signals(self, df_features: pd.DataFrame) -> TradingSignalDTO:
        """ML モデルで売買シグナルを生成する。

        元コードの GMOBot.predict_order() + get_buysell_signals() に相当。
        """
        buy_score = self._signal_buy.predict(df_features, self._feature_names)
        sell_score = self._signal_sell.predict(df_features, self._feature_names)
        logger.info(f"[{self.bot_id}] buy_score={buy_score:.4f} sell_score={sell_score:.4f}")

        # ダミー価格は呼び出し元が上書きする
        return TradingSignalDTO(
            buy_signal=buy_score > 0,
            sell_signal=sell_score > 0,
            buy_price=0,
            sell_price=0,
            buy_score=buy_score,
            sell_score=sell_score,
        )

    # ------------------------------------------------------------------ #
    # Entry / Exit                                                        #
    # ------------------------------------------------------------------ #

    def entry(self, df_features: pd.DataFrame, buy_price: int, sell_price: int) -> None:
        """新規エントリー注文を発注する。

        元コードの GMOBot.entry_position() に相当。
        """
        signal = self.compute_signals(df_features)

        # 既存ポジション量を集計
        buy_size = sum(
            p.size for p in self._position_state.values() if p.side == OrderSide.BUY
        )
        sell_size = sum(
            p.size for p in self._position_state.values() if p.side == OrderSide.SELL
        )

        if signal.buy_signal and buy_size == 0.0 and 0 < self.lot < 0.1:
            order_id = self._exchange.place_limit_order(
                self.symbol, OrderSide.BUY, self.lot, buy_price
            )
            if order_id:
                self._order_state[order_id] = ("BUY", self.lot, buy_price)
            logger.info(f"[{self.bot_id}] entry BUY {self.lot} @ {buy_price}")

        if signal.sell_signal and sell_size == 0.0 and 0 < self.lot < 0.1:
            order_id = self._exchange.place_limit_order(
                self.symbol, OrderSide.SELL, self.lot, sell_price
            )
            if order_id:
                self._order_state[order_id] = ("SELL", self.lot, sell_price)
            logger.info(f"[{self.bot_id}] entry SELL {self.lot} @ {sell_price}")

        self._persist_state()

    def exit_limit(self, buy_price: int, sell_price: int) -> None:
        """オープンポジションを指値で決済する。

        元コードの GMOBot.exit_position() に相当。
        """
        for pos_id, pos in self._position_state.items():
            if pos.side == OrderSide.BUY:
                order_id = self._exchange.place_limit_close_order(
                    self.symbol, OrderSide.SELL, pos.size, sell_price, pos_id
                )
                self._order_state[order_id] = ("CLOSE-BUY", pos.size, sell_price)
                logger.info(f"[{self.bot_id}] exit BUY pos={pos_id} @ {sell_price}")
            else:
                order_id = self._exchange.place_limit_close_order(
                    self.symbol, OrderSide.BUY, pos.size, buy_price, pos_id
                )
                self._order_state[order_id] = ("CLOSE-SELL", pos.size, buy_price)
                logger.info(f"[{self.bot_id}] exit SELL pos={pos_id} @ {buy_price}")

        self._persist_state()

    def exit_market(self) -> None:
        """オープンポジションをすべて成行で決済する。

        元コードの GMOBot.exit_position_market() に相当。
        """
        for pos_id, pos in self._position_state.items():
            close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            self._exchange.place_market_close_order(
                self.symbol, close_side, pos.size, pos_id
            )
            logger.info(f"[{self.bot_id}] exit_market {pos.side} pos={pos_id}")

    def run_cycle(self, df: pd.DataFrame, df_features: pd.DataFrame) -> None:
        """1サイクル分の exit → entry を実行する。

        元コードの GMOBot.exit_and_entry() に相当。
        """
        buy_price, sell_price = self.compute_order_prices(df)
        self.exit_limit(buy_price, sell_price)
        self.entry(df_features, buy_price, sell_price)

    def update_lot(self, new_lot: float) -> None:
        self.lot = new_lot

    # ------------------------------------------------------------------ #
    # Private                                                             #
    # ------------------------------------------------------------------ #

    def _persist_state(self) -> None:
        self._state.save_order_state(self.bot_id, self._order_state)
        self._state.save_position_state(self.bot_id, self._position_state)
