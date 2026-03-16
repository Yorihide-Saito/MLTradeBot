from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import List

from loguru import logger

from src.application.services.bot_agent import BotAgent
from src.application.services.lot_allocator import LotAllocator
from src.application.services.maintenance_scheduler import MaintenanceScheduler
from src.domain.entities.position import Position
from src.domain.ports.exchange_port import ExchangePort
from src.domain.ports.ohlcv_repository_port import OHLCVRepositoryPort
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.feature_engineering.feature_selector import FeatureSelector


class BotOrchestrator:
    """複数の BotAgent を管理し、15分足ごとの取引サイクルを実行する。

    元コードの Manager クラスを Clean Architecture に移植したもの。
    """

    def __init__(
        self,
        agents: List[BotAgent],
        exchange: ExchangePort,
        ohlcv_repo: OHLCVRepositoryPort,
        feature_calculator: TALibFeatureCalculator,
        feature_selector: FeatureSelector,
        lot_allocator: LotAllocator,
        maintenance_scheduler: MaintenanceScheduler,
        symbol: str,
        interval_minutes: int = 15,
    ) -> None:
        self._agents = agents
        self._exchange = exchange
        self._ohlcv_repo = ohlcv_repo
        self._feature_calc = feature_calculator
        self._feature_sel = feature_selector
        self._lot_alloc = lot_allocator
        self._maintenance = maintenance_scheduler
        self._symbol = symbol
        self._interval = interval_minutes

    # ------------------------------------------------------------------ #
    # Bulk operations across all agents                                   #
    # ------------------------------------------------------------------ #

    def sync_all_agents(self) -> None:
        """全ボットの状態を最新の約定履歴に同期する。"""
        executions = self._exchange.get_recent_executions(self._symbol, count=100)
        for agent in self._agents:
            agent.sync_state_from_executions(executions)

    def cancel_all_pending(self) -> None:
        """全ボットの保留注文をキャンセルし、取引所一括キャンセルも実行する。"""
        self.sync_all_agents()
        for agent in self._agents:
            agent.cancel_pending_orders()
        time.sleep(1.0)

        # 取引所レベルでの完全キャンセルを確認
        self._exchange.cancel_all_orders(self._symbol)
        while True:
            active = self._exchange.get_active_orders(self._symbol)
            if not active:
                break
            time.sleep(1.5)
            self._exchange.cancel_all_orders(self._symbol)
            time.sleep(1.5)

        self.sync_all_agents()
        for agent in self._agents:
            agent._order_state.clear()

    def close_all_market(self) -> None:
        """全ボットのポジションを成行で決済する (メンテナンス前)。"""
        for agent in self._agents:
            agent.exit_market()

    def update_lots(self, current_price: float) -> None:
        """証拠金残高に基づいて全ボットのロットを更新する。"""
        jpy = self._exchange.get_account_jpy_balance()
        lots = self._lot_alloc.allocate(jpy, current_price, len(self._agents))
        for agent, lot in zip(self._agents, lots):
            agent.update_lot(lot)
        logger.info(f"Lots updated: {lots}")

    def detect_and_close_untracked(self) -> None:
        """ボットが管理していないポジションを検出して成行決済する。"""
        try:
            summary = self._exchange.get_position_summary(self._symbol)
            all_tracked: dict[int, Position] = {}
            for agent in self._agents:
                all_tracked.update(agent._position_state)

            # 合計追跡量と取引所の実際量を比較
            tracked_buy = sum(p.size for p in all_tracked.values() if p.side.value == "BUY")
            tracked_sell = sum(p.size for p in all_tracked.values() if p.side.value == "SELL")

            untracked_buy = round(summary.buy_quantity - tracked_buy, 2)
            untracked_sell = round(summary.sell_quantity - tracked_sell, 2)

            from src.domain.entities.order import OrderSide
            if untracked_buy > 0:
                self._exchange.place_market_close_bulk_order(self._symbol, OrderSide.SELL, untracked_buy)
                logger.warning(f"Closed untracked BUY position: {untracked_buy}")
            if untracked_sell > 0:
                self._exchange.place_market_close_bulk_order(self._symbol, OrderSide.BUY, untracked_sell)
                logger.warning(f"Closed untracked SELL position: {untracked_sell}")
        except Exception:
            logger.error(traceback.format_exc())

    def run_trading_cycle(self, df, df_features) -> None:
        """全ボットで exit → entry を実行する。"""
        for agent in self._agents:
            logger.info(f"[{agent.bot_id}] starting cycle")
            try:
                agent.run_cycle(df, df_features)
            except Exception:
                logger.error(f"[{agent.bot_id}] cycle error:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------ #
    # Main event loop                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """メインの無限ループ。15分ごとに取引サイクルを実行する。

        元コードの Manager.start_all_bots() に相当。
        """
        logger.info("BotOrchestrator starting...")
        self.sync_all_agents()
        self.cancel_all_pending()

        df = self._ohlcv_repo.fetch_and_update(self._symbol, self._interval)
        df_features = self._calculate_features(df)
        self.update_lots(float(df["cl"].iloc[-1]))

        prev_minute = (datetime.now().minute // self._interval) * self._interval

        while True:
            dt_now = datetime.now()

            # ---- メンテナンス事前処理 ----
            if self._maintenance.is_pre_maintenance(dt_now):
                try:
                    logger.info("Pre-maintenance: closing all positions")
                    self.sync_all_agents()
                    self.cancel_all_pending()
                    self.close_all_market()
                    time.sleep(20)
                    self.sync_all_agents()
                    time.sleep(60)
                except Exception:
                    logger.error(traceback.format_exc())

            # ---- メンテナンス中はスキップ ----
            if self._maintenance.is_in_maintenance(dt_now):
                time.sleep(30)
                continue

            # ---- 通常取引 ----
            try:
                time.sleep(5.0)
                self.sync_all_agents()
                df = self._ohlcv_repo.fetch_and_update(self._symbol, self._interval)

                current_interval_minute = (dt_now.minute // self._interval) * self._interval
                bar_started = df.index[-1].minute == dt_now.minute
                new_bar = bar_started and current_interval_minute != prev_minute

                if new_bar:
                    prev_minute = current_interval_minute
                    logger.info(f"New {self._interval}m bar at {dt_now}")

                    self.cancel_all_pending()
                    self.update_lots(float(df["cl"].iloc[-1]))
                    df = self._ohlcv_repo.fetch_and_update(self._symbol, self._interval)
                    df_features = self._calculate_features(df)
                    self.detect_and_close_untracked()
                    self.run_trading_cycle(df, df_features)
                    time.sleep(70)

            except Exception:
                logger.error(traceback.format_exc())

    # ------------------------------------------------------------------ #
    # Private                                                             #
    # ------------------------------------------------------------------ #

    def _calculate_features(self, df):
        df_sliced = df[df.index > "2021-08-01"]
        return self._feature_calc.calculate(df_sliced)
