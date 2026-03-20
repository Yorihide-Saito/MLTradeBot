from __future__ import annotations

from dataclasses import dataclass

from mltradebot.domain.entities.order import OrderSide, SettleType


@dataclass(frozen=True)
class Position:
    position_id: int
    settle_type: SettleType  # always OPEN for tracked positions
    side: OrderSide
    size: float
    entry_price: int


@dataclass
class PositionSummary:
    buy_quantity: float = 0.0
    sell_quantity: float = 0.0
    buy_avg_rate: float = 0.0
    sell_avg_rate: float = 0.0
