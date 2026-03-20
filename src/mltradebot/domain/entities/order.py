from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class ExecutionType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class SettleType(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class Order:
    order_id: int
    symbol: str
    side: OrderSide
    execution_type: ExecutionType
    size: float
    price: Optional[int]  # None for MARKET orders
    status: OrderStatus
    created_at: datetime = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Execution:
    order_id: int
    position_id: int
    settle_type: SettleType
    side: OrderSide
    size: float
    price: float
    loss_gain: float
    timestamp: datetime
