"""
training/binance_data_fetcher.py — Binance BTCUSDT + USDJPY → BTCJPY 換算データ取得

時間合わせ戦略:
  - BTCUSDT: Binance REST API (GET /api/v3/klines, 認証不要, 1000本/req)
  - USDJPY:  Yahoo Finance 経由 yfinance (1h足)
  - 時刻統一: すべて UTC naive datetime として扱う
  - USDJPY を BTC の 15min インデックスに reindex → forward-fill (週末・祝日対応)
  - OHLC を USD/JPY レートで乗算; volume は USDT 建てのまま (相対値として学習に使用)

使い方:
  from training.binance_data_fetcher import BinanceDataFetcher
  df = BinanceDataFetcher().fetch_btcjpy(date(2024, 1, 1), date.today())
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import yfinance as yf
from loguru import logger


class BinanceDataFetcher:
    """Binance BTCUSDT を取得し USDJPY 換算した OHLCV を返す。"""

    _BASE_URL = "https://api.binance.com"
    _SYMBOL = "BTCUSDT"
    _USDJPY_TICKER = "USDJPY=X"
    _REQUEST_DELAY = 0.12  # Binance rate limit: ~10 req/s (余裕を持って)

    def fetch_btcjpy(
        self,
        start: date,
        end: date,
        interval: str = "15m",
    ) -> pd.DataFrame:
        """BTCJPY換算のOHLCVを返す。

        Args:
            start: 取得開始日 (含む)
            end:   取得終了日 (含まない、date.today() を渡す想定)
            interval: Binance interval 文字列 ('1m','5m','15m','1h','4h','1d')

        Returns:
            列 [op, hi, lo, cl, volume] / UTC naive DatetimeIndex の DataFrame
        """
        logger.info(f"BinanceFetcher: BTCUSDT {start} → {end} ({interval})")
        btc_df = self._fetch_btcusdt(start, end, interval)

        logger.info("BinanceFetcher: USDJPY from Yahoo Finance")
        usdjpy = self._fetch_usdjpy(start, end)

        result = self._convert_to_jpy(btc_df, usdjpy)
        logger.info(f"BinanceFetcher: {len(result)} rows ready (BTCJPY)")
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_btcusdt(self, start: date, end: date, interval: str) -> pd.DataFrame:
        start_ms = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end, datetime.min.time()).timestamp() * 1000)

        all_rows: list = []
        current_ms = start_ms

        with httpx.Client(timeout=30) as client:
            while current_ms < end_ms:
                resp = client.get(
                    f"{self._BASE_URL}/api/v3/klines",
                    params={
                        "symbol": self._SYMBOL,
                        "interval": interval,
                        "startTime": current_ms,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                )
                resp.raise_for_status()
                data: list = resp.json()

                if not data:
                    break

                all_rows.extend(data)
                current_ms = int(data[-1][0]) + 1  # 次のロウソク足

                if len(data) < 1000:
                    break  # 最終ページ

                time.sleep(self._REQUEST_DELAY)

        logger.info(f"Binance: {len(all_rows)} candles fetched")
        return self._parse_klines(all_rows)

    @staticmethod
    def _parse_klines(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time", "op", "hi", "lo", "cl", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("open_time")[["op", "hi", "lo", "cl", "volume"]].astype(float)
        df.index.name = None
        return df

    def _fetch_usdjpy(self, start: date, end: date) -> pd.Series:
        """Yahoo Finance から 1h 足 USDJPY を取得して UTC naive Series を返す。"""
        # 週末バッファとして前後 5 日多めに取得
        dl_start = start - timedelta(days=5)
        dl_end = end + timedelta(days=2)

        ticker = yf.download(
            self._USDJPY_TICKER,
            start=dl_start,
            end=dl_end,
            interval="1h",
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if ticker.empty:
            raise RuntimeError(
                "USDJPY データを Yahoo Finance から取得できませんでした。"
                "ネットワーク接続を確認してください。"
            )

        rate: pd.Series = ticker["Close"].squeeze()

        # timezone を UTC naive に統一
        if rate.index.tz is not None:
            rate.index = rate.index.tz_convert("UTC").tz_localize(None)

        return rate.dropna()

    @staticmethod
    def _convert_to_jpy(btc_df: pd.DataFrame, usdjpy: pd.Series) -> pd.DataFrame:
        """OHLC を USD/JPY レートで乗算して JPY 建てに換算する。

        時間合わせ:
          1. BTC インデックス + USDJPY インデックスの和集合で reindex
          2. forward-fill (週末・祝日の欠損を直前レートで補完)
          3. backward-fill (データ先頭部の欠損対応)
          4. BTC 15min インデックスに絞り込み
        """
        combined_idx = btc_df.index.union(usdjpy.index)
        rate_aligned = (
            usdjpy
            .reindex(combined_idx)
            .ffill()
            .bfill()
            .reindex(btc_df.index)
        )

        missing = rate_aligned.isna().sum()
        if missing > 0:
            logger.warning(f"USDJPY: {missing} 件のレートが補完できませんでした。除外します。")

        result = btc_df.copy()
        for col in ["op", "hi", "lo", "cl"]:
            result[col] = (result[col] * rate_aligned).round(0)

        # volume は USDT 建てのまま (モデルは相対的な増減を学習)
        return result.dropna()
