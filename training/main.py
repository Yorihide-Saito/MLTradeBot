"""
training/main.py — 再学習 CLI エントリーポイント

使い方:
  # Docker
  docker compose --profile retrain run retrain

  # ローカル (直近 30 日で再学習)
  python -m training.main --trials 50

  # CSV bootstrap 後に再学習 (長期データ推奨)
  python -m training.main --csv /work/data/BitFlyer_BTCJPY_1h.csv --trials 100

  # 特定ボットのみ
  python -m training.main --bot-id 0p186 --trials 50

BitFlyer の制約 (31 日のみ) への対処:
  https://www.cryptodatadownload.com/data/bitflyer/ から CSV をダウンロード後、
  --csv オプションで渡すとキャッシュとマージされる。
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Retrain trading models (BitFlyer)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history via REST (max 30; use --csv for longer)")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--bot-id", type=str, default=None, help="Retrain specific bot only")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV path for historical bootstrap (cryptodatadownload.com 形式)")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(level=settings.log_level, debug=settings.debug)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    pipeline, fetcher = build_pipeline(settings, n_trials=args.trials)

    if args.csv:
        logger.info(f"Importing CSV: {args.csv}")
        df = fetcher.import_csv_and_merge(Path(args.csv))
        logger.info(f"CSV merged: {len(df)} rows")

    logger.info(f"Starting retrain: days={args.days}, trials={args.trials}")

    if args.bot_id:
        model_repo = JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir)
        pipeline.run(args.bot_id, model_repo.get_atr_coeff(args.bot_id), days_of_history=args.days)
    else:
        pipeline.run_all(days_of_history=args.days)

    logger.info("Retraining complete!")


if __name__ == "__main__":
    main()
