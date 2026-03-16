"""
再学習エントリーポイント — BitFlyer のみ使用

使い方:
  # Docker
  docker compose --profile retrain run retrain

  # ローカル (31日以内のデータで再学習)
  python -m src.entrypoints.retrain.retrain_main --trials 50

  # CSV bootstrap 後に再学習 (推奨: 長期データ)
  python -m src.entrypoints.retrain.retrain_main --csv /work/data/bitflyer_btcjpy.csv --trials 100

BitFlyer の制限 (31日のみ) への対処:
  1. https://www.cryptodatadownload.com/data/bitflyer/ から CSV をダウンロード
  2. --csv オプションで渡す → 自動的に既存キャッシュとマージされる
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.config.logging_config import configure_logging
from src.config.settings import Settings
from src.entrypoints.retrain.data_fetcher import HistoricalDataFetcher
from src.entrypoints.retrain.model_evaluator import ModelEvaluator
from src.entrypoints.retrain.retrain_pipeline import RetrainPipeline
from src.infrastructure.exchange.bitflyer.bitflyer_auth import BitFlyerAuthenticator
from src.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider
from src.infrastructure.exchange.bitflyer.bitflyer_http_client import BitFlyerHttpClient
from src.infrastructure.feature_engineering.feature_selector import FeatureSelector
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.persistence.joblib_model_repository import JoblibModelRepository


def build_retrain_pipeline(settings: Settings, n_trials: int) -> tuple[RetrainPipeline, HistoricalDataFetcher]:
    # BitFlyer HTTP (認証なしでも公開エンドポイントにアクセス可能)
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
    feature_calc = TALibFeatureCalculator()
    feature_sel = FeatureSelector(settings.feature_pkl_path)
    model_repo = JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir)
    evaluator = ModelEvaluator()

    pipeline = RetrainPipeline(
        data_fetcher=fetcher,
        feature_calc=feature_calc,
        feature_sel=feature_sel,
        model_repo=model_repo,
        evaluator=evaluator,
        eval_dir=settings.work_dir / "eval",
        n_trials=n_trials,
    )
    return pipeline, fetcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain trading models (BitFlyer only)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history from REST API (max 30 due to BitFlyer limit)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per model")
    parser.add_argument("--bot-id", type=str, default=None,
                        help="Retrain specific bot ID only")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to historical CSV (from cryptodatadownload.com etc.) "
                             "to bootstrap data beyond 31 days")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(level=settings.log_level, debug=settings.debug)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    pipeline, fetcher = build_retrain_pipeline(settings, n_trials=args.trials)

    # CSV bootstrap (長期データの場合)
    if args.csv:
        csv_path = Path(args.csv)
        logger.info(f"Importing historical CSV: {csv_path}")
        df = fetcher.import_csv_and_merge(csv_path)
        logger.info(f"CSV merged: {len(df)} rows available")

    logger.info(f"Starting retrain: days={args.days}, trials={args.trials}")

    if args.bot_id:
        model_repo = JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir)
        atr_coeff = model_repo.get_atr_coeff(args.bot_id)
        pipeline.run(args.bot_id, atr_coeff, days_of_history=args.days)
    else:
        pipeline.run_all(days_of_history=args.days)

    logger.info("Retraining complete!")


if __name__ == "__main__":
    main()
