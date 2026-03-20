"""
training/data_fetcher.py — BitFlyer 約定履歴から OHLCV 取得

BitFlyer の制約: REST API は過去 31 日のみ。
31 日以上の過去データが必要な場合は import_csv_and_merge() を使用。

CSV bootstrap 手順:
  1. https://www.cryptodatadownload.com/data/bitflyer/ から CSV をダウンロード
  2. python -m mltradebot.training.main --csv /path/to/file.csv
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from mltradebot.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider


class HistoricalDataFetcher:
    def __init__(self, provider: BitFlyerDataProvider) -> None:
        self._provider = provider

    def fetch_range(
        self,
        start_date: date,
        end_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        days_back = (date.today() - start_date).days
        if days_back > 30:
            logger.warning(
                f"BitFlyer は 31 日以上の約定履歴を提供しません。"
                f"長期データは --csv オプションで CSV を渡してください。\n"
                f"推奨ソース: https://www.cryptodatadownload.com/data/bitflyer/"
            )
            days_back = 30
        return self._provider._fetch_executions_to_ohlcv(days_back=days_back)

    def fetch_and_save(self, output_path: Path, days_back: int, interval_minutes: int) -> pd.DataFrame:
        start = date.today() - timedelta(days=min(days_back, 30))
        df = self.fetch_range(start, date.today(), interval_minutes)
        df.to_pickle(output_path)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return df

    def import_csv_and_merge(self, csv_path: Path) -> pd.DataFrame:
        """CSV から長期データを読み込んでキャッシュとマージする。"""
        return self._provider.import_csv(csv_path)
