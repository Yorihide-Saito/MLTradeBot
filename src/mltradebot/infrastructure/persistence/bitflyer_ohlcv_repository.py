from __future__ import annotations

from pathlib import Path

import pandas as pd

from mltradebot.domain.entities.candle import OHLCVSeries
from mltradebot.domain.ports.ohlcv_repository_port import OHLCVRepositoryPort
from mltradebot.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider


class BitFlyerOHLCVRepository(OHLCVRepositoryPort):
    """BitFlyer の実行履歴から OHLCV を構築・キャッシュする Repository。

    GMO Coin の PickleOHLCVRepository の BitFlyer 版。
    データは BitFlyerDataProvider が担い、このクラスは OHLCVRepositoryPort を満たす。
    """

    def __init__(self, provider: BitFlyerDataProvider) -> None:
        self._provider = provider

    def load(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        return self._provider.load()

    def save(self, df: OHLCVSeries, symbol: str, interval_minutes: int) -> None:
        self._provider.save(df)

    def fetch_and_update(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        # WebSocket バッファがあればフラッシュして REST で補完
        df_ws = self._provider.flush_websocket_buffer()
        if not df_ws.empty:
            return df_ws
        return self._provider.fetch_and_update()
