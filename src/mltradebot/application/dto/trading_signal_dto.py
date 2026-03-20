from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingSignalDTO:
    """ボットの売買シグナルと注文価格をまとめた DTO。"""

    buy_signal: bool
    sell_signal: bool
    buy_price: int
    sell_price: int
    buy_score: float = 0.0
    sell_score: float = 0.0
