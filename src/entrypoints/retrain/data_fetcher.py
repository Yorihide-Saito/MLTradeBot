"""
BitFlyer のみを使った歴史データ収集。

BitFlyer の制約:
- OHLCV API なし
- getexecutions は過去 31 日のみ
- 31日以上の過去データは CSV から bootstrap する

再学習で長期データが必要な場合:
  1. cryptodatadownload.com 等から BTC/JPY CSV をダウンロード
  2. BitFlyerDataProvider.import_csv() で読み込む
  3. 以後は WebSocket + REST で差分を積み上げる
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider


class HistoricalDataFetcher:
    """BitFlyer の約定履歴から OHLCV の歴史データを取得する。

    - 過去 31 日以内: REST /v1/getexecutions + cursor pagination
    - 31 日以上: CSV インポートで bootstrap (BitFlyer API の制限)
    """

    def __init__(self, provider: BitFlyerDataProvider) -> None:
        self._provider = provider

    def fetch_range(
        self,
        start_date: date,
        end_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        """指定期間の OHLCV を取得して返す。

        BitFlyer の 31 日制限のため、start_date が 31 日以上前の場合は
        警告を出し、取得できる範囲だけ返す。
        """
        days_back = (date.today() - start_date).days
        if days_back > 30:
            logger.warning(
                f"BitFlyer only provides 31 days of execution history. "
                f"Requested {days_back} days. "
                f"For longer history, use import_csv() with a downloaded CSV.\n"
                f"Suggested source: https://www.cryptodatadownload.com/data/bitflyer/"
            )
            days_back = 30

        return self._provider._fetch_executions_to_ohlcv(days_back=days_back)

    def fetch_and_save(
        self,
        output_path: Path,
        days_back: int,
        interval_minutes: int,
    ) -> pd.DataFrame:
        """過去 days_back 日間のデータを取得して pickle 保存する。"""
        start = date.today() - timedelta(days=min(days_back, 30))
        end = date.today()
        df = self.fetch_range(start, end, interval_minutes)
        df.to_pickle(output_path)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return df

    def import_csv_and_merge(self, csv_path: Path) -> pd.DataFrame:
        """CSV から長期データを読み込んでキャッシュにマージする。

        使い方:
          1. https://www.cryptodatadownload.com/data/bitflyer/ から
             「BitFlyer BTCJPY 1 hour」等をダウンロード
          2. このメソッドで読み込む (1h CSVを15分足にresampleする)
        """
        return self._provider.import_csv(csv_path)
