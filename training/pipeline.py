"""
training/pipeline.py — 再学習パイプライン統合

ステップ:
  1. BitFlyer から OHLCV 取得 (REST 31日 or CSV bootstrap)
  2. TA-Lib で特徴量計算
  3. ATR ベースラベル生成
  4. Optuna + Walk-forward CV で LightGBM 学習
  5. 評価レポート出力
  6. モデル保存 (model_buy/ model_sell/)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.infrastructure.feature_engineering.feature_selector import FeatureSelector
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.persistence.joblib_model_repository import JoblibModelRepository
from training.data_fetcher import HistoricalDataFetcher
from training.label_generator import LabelGenerator
from training.model_evaluator import ModelEvaluator
from training.model_trainer import ModelTrainer


class RetrainPipeline:
    def __init__(
        self,
        data_fetcher: HistoricalDataFetcher,
        feature_calc: TALibFeatureCalculator,
        feature_sel: FeatureSelector,
        model_repo: JoblibModelRepository,
        evaluator: ModelEvaluator,
        eval_dir: Path,
        atr_period: int = 14,
        forward_bars: int = 4,
        n_trials: int = 50,
    ) -> None:
        self._fetcher = data_fetcher
        self._feature_calc = feature_calc
        self._feature_sel = feature_sel
        self._model_repo = model_repo
        self._evaluator = evaluator
        self._eval_dir = eval_dir
        self._atr_period = atr_period
        self._forward_bars = forward_bars
        self._n_trials = n_trials

    def run(
        self,
        bot_id: str,
        atr_coeff: float,
        days_of_history: int = 30,
        df_raw: pd.DataFrame | None = None,
    ) -> None:
        """1 bot_id のモデルペアを再学習する。

        Args:
            df_raw: 外部から渡す OHLCV DataFrame。
                    None の場合は BitFlyer REST から取得 (最大 30 日)。
                    Binance など外部ソースを使う場合は呼び出し元で渡す。
        """
        logger.info(f"[{bot_id}] Retraining: atr_coeff={atr_coeff}")

        # 1. データ取得
        if df_raw is None:
            end = date.today()
            start = end - timedelta(days=min(days_of_history, 30))
            df_raw = self._fetcher.fetch_range(start, end, interval_minutes=15)

        if len(df_raw) < 500:
            logger.warning(f"[{bot_id}] データ不足 ({len(df_raw)} rows)。CSV bootstrap を推奨。")
            return

        # 2. 特徴量計算
        df_feat = self._feature_calc.calculate(df_raw)
        feature_names = self._feature_sel.load_feature_names()

        # 3. ラベル生成
        df_labeled = LabelGenerator(atr_coeff, self._atr_period, self._forward_bars).generate_both(df_feat)
        df_labeled = df_labeled.dropna(subset=feature_names)

        # 4. 学習
        trainer = ModelTrainer(feature_names=feature_names, n_splits=5, n_trials=self._n_trials)
        logger.info(f"[{bot_id}] Training BUY model...")
        model_buy, params_buy = trainer.train(df_labeled, "y_buy")
        logger.info(f"[{bot_id}] Training SELL model...")
        model_sell, params_sell = trainer.train(df_labeled, "y_sell")

        # 5. 評価
        run_id = f"{bot_id}_{date.today()}"
        metrics = {
            "buy": self._evaluator.evaluate(model_buy, df_labeled, "y_buy", feature_names),
            "sell": self._evaluator.evaluate(model_sell, df_labeled, "y_sell", feature_names),
            "params_buy": params_buy,
            "params_sell": params_sell,
        }
        self._evaluator.save_report(metrics, self._eval_dir, run_id)

        # 6. モデル保存
        self._model_repo.save_model_pair(bot_id, model_buy, model_sell)
        logger.info(f"[{bot_id}] Models saved.")

    def run_all(
        self,
        days_of_history: int = 30,
        df_raw: pd.DataFrame | None = None,
    ) -> None:
        """全 bot_id を再学習する。df_raw を渡すと全 bot で同じデータを共用する。"""
        for bot_id in self._model_repo.list_bot_ids():
            atr_coeff = self._model_repo.get_atr_coeff(bot_id)
            try:
                self.run(bot_id, atr_coeff, days_of_history, df_raw=df_raw)
            except Exception as e:
                logger.error(f"[{bot_id}] Failed: {e}")
