from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from mltradebot.domain.entities.candle import OHLCVSeries
from mltradebot.domain.ports.ohlcv_repository_port import OHLCVRepositoryPort
from mltradebot.infrastructure.exchange.gmo.gmo_http_client import GMOHttpClient


class PickleOHLCVRepository(OHLCVRepositoryPort):
    """OHLCVデータを Pickle ファイルにキャッシュし、不足分を GMO 公開 API から補完する。

    元コードの update_ohlcv / get_gmo_ohlcv / get_latest_ohlcv の
    ロジックをここに集約する。
    """

    def __init__(self, data_dir: Path, http: GMOHttpClient) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._http = http

    # ------------------------------------------------------------------ #
    # OHLCVRepositoryPort                                                  #
    # ------------------------------------------------------------------ #

    def load(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        path = self._pkl_path(symbol, interval_minutes)
        if path.exists():
            return pd.read_pickle(path)
        return pd.DataFrame(columns=["op", "hi", "lo", "cl", "volume"])

    def save(self, df: OHLCVSeries, symbol: str, interval_minutes: int) -> None:
        df.to_pickle(self._pkl_path(symbol, interval_minutes))

    def fetch_and_update(self, symbol: str, interval_minutes: int) -> OHLCVSeries:
        df = self.load(symbol, interval_minutes)

        if df.empty:
            delta_days = 30  # 初回は直近30日を取得
        else:
            delta_days = (datetime.now() - df.index[-1]).days + 1

        df_latest = self._fetch_range(symbol, interval_minutes, delta_days)

        if df.empty:
            merged = df_latest
        else:
            # 重複を除いて結合
            new_rows = df_latest[~df_latest.index.isin(df.index)]
            merged = pd.concat([df, new_rows]).sort_index()

        # 最終足（形成中）は除外して保存
        self.save(merged[:-1], symbol, interval_minutes)
        logger.info(f"OHLCV updated: {symbol} {interval_minutes}m, rows={len(merged)}")
        return merged

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_range(
        self, symbol: str, interval_minutes: int, days_back: int
    ) -> OHLCVSeries:
        today = datetime.today()
        frames = []

        for i in range(days_back, 0, -1):
            day = (today - timedelta(days=i)).strftime("%Y%m%d")
            try:
                df_day = self._fetch_single_day(symbol, interval_minutes, day)
                frames.append(df_day)
            except Exception as e:
                logger.warning(f"Failed to fetch {day}: {e}")
            if i > 1:
                time.sleep(1.1)  # GMO レート制限

        # 当日分
        today_str = today.strftime("%Y%m%d")
        if datetime.now().hour >= 6:
            try:
                frames.append(self._fetch_single_day(symbol, interval_minutes, today_str))
            except Exception as e:
                logger.warning(f"Failed to fetch today {today_str}: {e}")

        if not frames:
            return pd.DataFrame(columns=["op", "hi", "lo", "cl", "volume"])

        return pd.concat(frames)

    def _fetch_single_day(
        self, symbol: str, interval_minutes: int, day: str
    ) -> OHLCVSeries:
        path = f"/v1/klines"
        params = {"symbol": symbol, "interval": f"{interval_minutes}min", "date": day}
        data = self._http.get_public(path, params)

        df = pd.DataFrame(data["data"])
        df.rename(
            columns={"openTime": "timestamp", "open": "op", "high": "hi", "low": "lo", "close": "cl"},
            inplace=True,
        )
        for col in ["op", "hi", "lo", "cl", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def _pkl_path(self, symbol: str, interval_minutes: int) -> Path:
        return self._data_dir / f"ohlcv_{symbol.lower()}_{interval_minutes}m.pkl"
