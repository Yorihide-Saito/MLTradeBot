"""
training/main.py — 再学習 CLI エントリーポイント

使い方:
  # Binance から 1 年分取得して再学習 (推奨)
  python -m training.main --binance --days 365 --trials 100

  # Binance + 特定 bot のみ
  python -m training.main --binance --days 365 --bot-id 0p186 --trials 50

  # Docker
  docker compose --profile retrain run retrain

  # BitFlyer REST のみ (直近 30 日)
  python -m training.main --trials 50

  # CSV bootstrap 後に再学習 (BitFlyer 用長期データ)
  python -m training.main --csv /work/data/BitFlyer_BTCJPY_1h.csv --trials 100

BitFlyer の制約 (31 日のみ) への対処:
  --binance を使うと Binance BTCUSDT を USDJPY 換算して取得する。
  認証不要。yfinance (USDJPY=X) で為替レートを補完する。
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

from src.config.logging_config import configure_logging
from src.config.settings import Settings
from src.infrastructure.exchange.bitflyer.bitflyer_auth import BitFlyerAuthenticator
from src.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider
from src.infrastructure.exchange.bitflyer.bitflyer_http_client import BitFlyerHttpClient
from src.infrastructure.feature_engineering.feature_selector import FeatureSelector
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.persistence.joblib_model_repository import JoblibModelRepository
from training.data_fetcher import HistoricalDataFetcher
from training.model_evaluator import ModelEvaluator
from training.pipeline import RetrainPipeline


def build_pipeline(settings: Settings, n_trials: int) -> tuple[RetrainPipeline, HistoricalDataFetcher]:
    auth = BitFlyerAuthenticator(
        settings.bitflyer_api_key or "",
        settings.bitflyer_secret_key or "",
    )
    http = BitFlyerHttpClient(auth)
    provider = BitFlyerDataProvider(
        http=http,
        data_dir=settings.data_dir,
        product_code=settings.symbol,
        interval_minutes=settings.candle_interval_minutes,
    )
    fetcher = HistoricalDataFetcher(provider)
    pipeline = RetrainPipeline(
        data_fetcher=fetcher,
        feature_calc=TALibFeatureCalculator(),
        feature_sel=FeatureSelector(settings.feature_pkl_path),
        model_repo=JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir),
        evaluator=ModelEvaluator(),
        eval_dir=settings.work_dir / "eval",
        n_trials=n_trials,
    )
    return pipeline, fetcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain trading models")
    parser.add_argument(
        "--binance", action="store_true",
        help="Binance BTCUSDT + USDJPY 換算でデータ取得 (推奨: 長期学習用)",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="取得日数 (--binance 時は最大 730 日; BitFlyer 時は最大 30 日)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--bot-id", type=str, default=None, help="特定 bot のみ再学習")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="CSV パス (cryptodatadownload.com 形式; BitFlyer 用長期 bootstrap)",
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logging(level=settings.log_level, debug=settings.debug)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    pipeline, fetcher = build_pipeline(settings, n_trials=args.trials)
    model_repo = JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir)

    # ------------------------------------------------------------------ #
    # データソース選択                                                      #
    # ------------------------------------------------------------------ #

    df_raw = None  # None のとき pipeline 内で BitFlyer REST を使用

    if args.binance:
        from training.binance_data_fetcher import BinanceDataFetcher
        max_days = min(args.days, 730)  # yfinance の 1h 足上限
        start = date.today() - timedelta(days=max_days)
        logger.info(f"Binance モード: {start} → {date.today()} ({max_days} 日)")
        df_raw = BinanceDataFetcher().fetch_btcjpy(start=start, end=date.today())
        logger.info(f"取得完了: {len(df_raw)} rows")

    elif args.csv:
        logger.info(f"CSV インポート: {args.csv}")
        df_merged = fetcher.import_csv_and_merge(Path(args.csv))
        logger.info(f"CSV マージ完了: {len(df_merged)} rows")
        # CSV マージ後は BitFlyer REST キャッシュを pipeline が読む (df_raw=None のまま)

    # ------------------------------------------------------------------ #
    # 再学習実行                                                           #
    # ------------------------------------------------------------------ #

    logger.info(f"再学習開始: trials={args.trials}, bot_id={args.bot_id or 'all'}")

    if args.bot_id:
        pipeline.run(
            args.bot_id,
            model_repo.get_atr_coeff(args.bot_id),
            days_of_history=args.days,
            df_raw=df_raw,
        )
    else:
        pipeline.run_all(days_of_history=args.days, df_raw=df_raw)

    logger.info("再学習完了!")


if __name__ == "__main__":
    main()
