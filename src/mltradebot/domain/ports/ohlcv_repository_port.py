from __future__ import annotations

from abc import ABC, abstractmethod

from mltradebot.domain.entities.candle import OHLCVSeries


class OHLCVRepositoryPort(ABC):
    """OHLCVデータの読み書き・取得の抽象インターフェース。"""

    @abstractmethod
    def load(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        """ローカルキャッシュからOHLCVを読み込む。"""

    @abstractmethod
    def save(self, df: OHLCVSeries, symbol: str, interval_minutes: int) -> None:
        """OHLCVをローカルキャッシュに保存する。"""

    @abstractmethod
    def fetch_and_update(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        """不足分を取引所APIから取得し、キャッシュを更新して返す。"""
