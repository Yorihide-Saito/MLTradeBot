from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from mltradebot.domain.entities.position import Position


# orderId_dict の値の型: (side_str, size, price) のタプル
OrderStateValue = Tuple[str, float, int]


class StateRepositoryPort(ABC):
    """各ボットの注文・ポジション状態の永続化インターフェース。"""

    @abstractmethod
    def load_order_state(self, bot_id: str) -> Dict[int, OrderStateValue]:
        """注文ID -> (side, size, price) のマップを返す。"""

    @abstractmethod
    def save_order_state(self, bot_id: str, state: Dict[int, OrderStateValue]) -> None:
        """注文状態を保存する。"""

    @abstractmethod
    def load_position_state(self, bot_id: str) -> Dict[int, Position]:
        """ポジションID -> Position のマップを返す。"""

    @abstractmethod
    def save_position_state(self, bot_id: str, state: Dict[int, Position]) -> None:
        """ポジション状態を保存する。"""
