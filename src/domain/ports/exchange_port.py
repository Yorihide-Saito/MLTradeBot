from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from src.domain.entities.order import Execution, Order, OrderSide
from src.domain.entities.position import Position, PositionSummary


class ExchangePort(ABC):
    """取引所操作の抽象インターフェース。GMO Coin / BitFlyer 等を差し替え可能にする。"""

    # ---- Orders ----

    @abstractmethod
    def place_limit_order(self, symbol: str, side: OrderSide, size: float, price: int) -> int:
        """指値注文を発注し order_id を返す。"""

    @abstractmethod
    def place_market_order(self, symbol: str, side: OrderSide, size: float) -> int:
        """成行注文を発注し order_id を返す。"""

    @abstractmethod
    def place_limit_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
        position_id: int,
    ) -> int:
        """ポジション指定の指値決済注文を発注し order_id を返す。"""

    @abstractmethod
    def place_market_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        position_id: int,
    ) -> None:
        """ポジション指定の成行決済注文を発注する。"""

    @abstractmethod
    def place_limit_close_bulk_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
    ) -> int:
        """一括指値決済注文を発注し order_id を返す。"""

    @abstractmethod
    def place_market_close_bulk_order(self, symbol: str, side: OrderSide, size: float) -> None:
        """一括成行決済注文を発注する。"""

    # ---- Cancellations ----

    @abstractmethod
    def cancel_orders(self, order_ids: List[int]) -> None:
        """複数注文をキャンセルする。"""

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None:
        """指定シンボルの全注文をキャンセルする。"""

    # ---- Queries ----

    @abstractmethod
    def get_active_orders(self, symbol: str) -> List[Order]:
        """有効な注文一覧を取得する。"""

    @abstractmethod
    def get_position_summary(self, symbol: str) -> PositionSummary:
        """ポジションサマリーを取得する。"""

    @abstractmethod
    def get_open_positions(self, symbol: str) -> Dict[int, Position]:
        """オープンポジション {position_id: Position} を取得する。"""

    @abstractmethod
    def get_account_jpy_balance(self) -> int:
        """JPY 建て口座残高を取得する。"""

    @abstractmethod
    def get_recent_executions(self, symbol: str, count: int = 100) -> List[Execution]:
        """直近の約定履歴を取得する。"""
