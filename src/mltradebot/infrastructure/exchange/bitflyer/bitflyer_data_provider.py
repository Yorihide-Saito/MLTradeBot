"""
BitFlyer のみを使った OHLCV データプロバイダー。

BitFlyer にはネイティブ OHLCV API が存在しないため、
2段構えで OHLCV を構築する:

1. REST /v1/getexecutions  → 過去 31 日分の約定を取得してローソク足に集計
2. WebSocket lightning_executions_BTC_JPY → リアルタイムで随時更新

長期の歴史データが必要な場合 (再学習など) は、
CSV インポートメソッドで bootstrap する。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from mltradebot.infrastructure.exchange.bitflyer.bitflyer_http_client import BitFlyerHttpClient


class BitFlyerDataProvider:
    """BitFlyer の実行履歴・WebSocket から OHLCV を構築・保存する。"""

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        http: BitFlyerHttpClient,
        data_dir: Path,
        product_code: str = "BTC_JPY",
        interval_minutes: int = 15,
    ) -> None:
        self._http = http
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._product = product_code
        self._interval = interval_minutes
        self._pkl_path = data_dir / f"ohlcv_{product_code.lower()}_{interval_minutes}m.pkl"

        # WebSocket 用バッファ (tick → candle 集計)
        self._ws_buffer: list[dict] = []
        self._ws_lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Public: OHLCV load / update                                         #
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """ローカルキャッシュから OHLCV を返す。"""
        if self._pkl_path.exists():
            return pd.read_pickle(self._pkl_path)
        return pd.DataFrame(columns=["op", "hi", "lo", "cl", "volume"])

    def save(self, df: pd.DataFrame) -> None:
        df.to_pickle(self._pkl_path)

    def fetch_and_update(self) -> pd.DataFrame:
        """REST API で過去 31 日分を取得してキャッシュを更新する。"""
        df_cached = self.load()

        if df_cached.empty:
            # 初回: 可能な限り遡って取得 (~31日)
            df_new = self._fetch_executions_to_ohlcv(days_back=30)
        else:
            # 差分取得: 最終足から現在まで
            delta_hours = (datetime.now(timezone.utc) - df_cached.index[-1].tz_localize("UTC")).seconds // 3600 + 1
            days_back = max(1, delta_hours // 24 + 1)
            df_new = self._fetch_executions_to_ohlcv(days_back=days_back)

        if df_new.empty:
            return df_cached

        if df_cached.empty:
            merged = df_new
        else:
            new_rows = df_new[~df_new.index.isin(df_cached.index)]
            merged = pd.concat([df_cached, new_rows]).sort_index()

        # 形成中の最終足は除外
        self.save(merged[:-1])
        logger.info(f"OHLCV updated: {self._product} {self._interval}m, rows={len(merged)}")
        return merged

    # ------------------------------------------------------------------ #
    # Public: CSV bootstrap (長期歴史データのインポート)                   #
    # ------------------------------------------------------------------ #

    def import_csv(self, csv_path: Path) -> pd.DataFrame:
        """外部 CSV から OHLCV を読み込み、キャッシュとマージする。

        対応フォーマット:
          - Cryptodatadownload.com 形式 (unix, date, symbol, open, high, low, close, volume)
          - 汎用 OHLCV 形式 (timestamp/datetime index + open/high/low/close/volume)

        使い方:
          1. https://www.cryptodatadownload.com/ などから
             BitFlyer BTC_JPY の CSV をダウンロード
          2. provider.import_csv(Path("BitFlyer_BTCJPY_1h.csv")) を実行
          3. ローカルキャッシュにマージされる
        """
        logger.info(f"Importing CSV: {csv_path}")
        raw = pd.read_csv(csv_path, skiprows=1)  # CDD は1行目にライセンス行

        # Cryptodatadownload.com フォーマットの検出
        if "unix" in raw.columns:
            raw = self._parse_cryptodatadownload(raw)
        else:
            raw = self._parse_generic_csv(raw)

        # 足のインターバルに resample
        df_resampled = self._resample(raw, self._interval)

        # 既存キャッシュとマージ
        df_cached = self.load()
        if df_cached.empty:
            merged = df_resampled
        else:
            new_rows = df_resampled[~df_resampled.index.isin(df_cached.index)]
            merged = pd.concat([df_resampled, new_rows]).sort_index()

        self.save(merged)
        logger.info(f"CSV imported: {len(merged)} rows total")
        return merged

    # ------------------------------------------------------------------ #
    # Public: WebSocket real-time accumulation                            #
    # ------------------------------------------------------------------ #

    def start_websocket(self) -> None:
        """WebSocket で約定データをバックグラウンド収集する。"""
        try:
            import websocket as ws_lib  # websocket-client ライブラリ
        except ImportError:
            logger.warning("websocket-client not installed. Run: pip install websocket-client")
            return

        def _on_message(ws, message: str) -> None:
            data = json.loads(message)
            if data.get("method") == "channelMessage":
                execs = data["params"]["message"]
                if isinstance(execs, list):
                    with self._ws_lock:
                        self._ws_buffer.extend(execs)

        def _on_open(ws) -> None:
            subscribe_msg = json.dumps({
                "method": "subscribe",
                "params": {"channel": f"lightning_executions_{self._product}"},
                "id": 1,
            })
            ws.send(subscribe_msg)
            logger.info(f"WebSocket subscribed: lightning_executions_{self._product}")

        def _run() -> None:
            while True:
                try:
                    ws = ws_lib.WebSocketApp(
                        "wss://ws.lightstream.bitflyer.com/json-rpc",
                        on_message=_on_message,
                        on_open=_on_open,
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}, reconnecting in 5s")
                    time.sleep(5)

        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()

    def flush_websocket_buffer(self) -> pd.DataFrame:
        """WebSocket バッファを OHLCV にフラッシュしてキャッシュを更新する。"""
        with self._ws_lock:
            if not self._ws_buffer:
                return self.load()
            buffer = self._ws_buffer.copy()
            self._ws_buffer.clear()

        df_ticks = pd.DataFrame(buffer)
        df_ticks["exec_date"] = pd.to_datetime(df_ticks["exec_date"])
        df_ticks = df_ticks.set_index("exec_date").sort_index()
        df_ticks["price"] = df_ticks["price"].astype(float)
        df_ticks["size"] = df_ticks["size"].astype(float)

        df_new_candles = self._resample(df_ticks.rename(columns={"price": "cl"}), self._interval)

        df_cached = self.load()
        if df_cached.empty:
            merged = df_new_candles
        else:
            new_rows = df_new_candles[~df_new_candles.index.isin(df_cached.index)]
            merged = pd.concat([df_cached, new_rows]).sort_index()

        self.save(merged[:-1])  # 形成中の最終足は除外
        return merged

    # ------------------------------------------------------------------ #
    # Private: REST execution → OHLCV                                     #
    # ------------------------------------------------------------------ #

    def _fetch_executions_to_ohlcv(self, days_back: int) -> pd.DataFrame:
        """REST getexecutions を cursor ページングで取得し OHLCV に集計する。"""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        all_execs: list[dict] = []
        before_id: Optional[int] = None

        logger.info(f"Fetching executions since {since} (up to 31 days)")

        while True:
            params: dict = {"product_code": self._product, "count": 500}
            if before_id is not None:
                params["before"] = before_id

            try:
                execs = self._http.get_public(f"/v1/getexecutions", params)
            except Exception as e:
                logger.error(f"Execution fetch error: {e}")
                break

            if not execs:
                break

            all_execs.extend(execs)
            oldest_ts = datetime.fromisoformat(execs[-1]["exec_date"])
            if oldest_ts.tzinfo is None:
                oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)

            if oldest_ts < since:
                break

            before_id = int(execs[-1]["id"])
            time.sleep(0.3)  # レート制限: 500req/5min

        if not all_execs:
            return pd.DataFrame(columns=["op", "hi", "lo", "cl", "volume"])

        df = pd.DataFrame(all_execs)
        df["exec_date"] = pd.to_datetime(df["exec_date"]).dt.tz_localize("UTC")
        df["price"] = df["price"].astype(float)
        df["size"] = df["size"].astype(float)
        df = df.set_index("exec_date").sort_index()
        df = df[df.index >= since]

        return self._resample(df, self._interval)

    @staticmethod
    def _resample(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
        """tick DataFrame を OHLCV ローソク足に集計する。"""
        price_col = "price" if "price" in df.columns else "cl"
        size_col = "size" if "size" in df.columns else "volume"

        rule = f"{interval_minutes}min"
        ohlcv = df[price_col].resample(rule).ohlc()
        ohlcv.columns = ["op", "hi", "lo", "cl"]
        ohlcv["volume"] = df[size_col].resample(rule).sum()
        ohlcv = ohlcv.dropna()
        return ohlcv

    @staticmethod
    def _parse_cryptodatadownload(raw: pd.DataFrame) -> pd.DataFrame:
        """Cryptodatadownload.com 形式のCSVをパースする。"""
        df = raw.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["unix"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        rename = {"open": "op", "high": "hi", "low": "lo", "close": "cl", "volume btc": "volume"}
        df = df.rename(columns=rename)
        for col in ["op", "hi", "lo", "cl", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["op", "hi", "lo", "cl", "volume"]].dropna()

    @staticmethod
    def _parse_generic_csv(raw: pd.DataFrame) -> pd.DataFrame:
        """汎用 OHLCV CSV をパースする。"""
        df = raw.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), df.columns[0])
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index("timestamp").sort_index()
        rename = {"open": "op", "high": "hi", "low": "lo", "close": "cl"}
        df = df.rename(columns=rename)
        for col in ["op", "hi", "lo", "cl", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["op", "hi", "lo", "cl", "volume"]].dropna()
