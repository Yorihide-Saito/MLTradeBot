from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

from loguru import logger

from src.entrypoints.retrain.data_fetcher import HistoricalDataFetcher
from src.entrypoints.retrain.label_generator import LabelGenerator
from src.entrypoints.retrain.model_evaluator import ModelEvaluator
from src.entrypoints.retrain.model_trainer import ModelTrainer
from src.infrastructure.feature_engineering.feature_selector import FeatureSelector
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.persistence.joblib_model_repository import JoblibModelRepository


class RetrainPipeline:
    """1 ボット分のモデルを再学習する完全なパイプライン。

    ステップ:
    1. 過去データを BitFlyer 約定履歴 (or CSV) から取得
    2. TA-Lib で特徴量を計算 (既存モデルと同じ特徴量)
    3. ATR ベースのラベルを生成
    4. Optuna + Walk-forward CV で LightGBM を学習
    5. 評価レポートを出力
    6. モデルを保存
    """

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

    def run(self, bot_id: str, atr_coeff: float, days_of_history: int = 400) -> None:
        """1 bot_id のモデルペアを再学習する。"""
        logger.info(f"[{bot_id}] Retraining start: atr_coeff={atr_coeff}, days={days_of_history}")

        # 1. データ取得
        # BitFlyer の制限: REST は 31 日のみ。
        # 長期データは retrain_main.py の --csv オプションで事前にキャッシュへ投入すること。
        end = date.today()
        start = end - timedelta(days=min(days_of_history, 30))
        df_raw = self._fetcher.fetch_range(start, end, interval_minutes=15)

        if len(df_raw) < 500:
            logger.warning(f"[{bot_id}] Not enough data ({len(df_raw)} rows), skipping")
            return

        # 2. 特徴量計算
        df_feat = self._feature_calc.calculate(df_raw)
        feature_names = self._feature_sel.load_feature_names()

        # 3. ラベル生成
        label_gen = LabelGenerator(atr_coeff, self._atr_period, self._forward_bars)
        df_labeled = label_gen.generate_both(df_feat)
        df_labeled = df_labeled.dropna(subset=feature_names)

        # 4. 学習
        trainer = ModelTrainer(
            feature_names=feature_names,
            n_splits=5,
            n_trials=self._n_trials,
        )

        logger.info(f"[{bot_id}] Training BUY model...")
        model_buy, params_buy = trainer.train(df_labeled, "y_buy")

        logger.info(f"[{bot_id}] Training SELL model...")
        model_sell, params_sell = trainer.train(df_labeled, "y_sell")

        # 5. 評価
        run_id = f"{bot_id}_{date.today()}"
        metrics_buy = self._evaluator.evaluate(model_buy, df_labeled, "y_buy", feature_names)
        metrics_sell = self._evaluator.evaluate(model_sell, df_labeled, "y_sell", feature_names)
        self._evaluator.save_report(
            {"buy": metrics_buy, "sell": metrics_sell, "params_buy": params_buy, "params_sell": params_sell},
            self._eval_dir,
            run_id,
        )

        # 6. モデル保存
        self._model_repo.save_model_pair(bot_id, model_buy, model_sell)
        logger.info(f"[{bot_id}] Models saved successfully")

    def run_all(self, days_of_history: int = 400) -> None:
        """全 bot_id のモデルを再学習する。"""
        bot_ids = self._model_repo.list_bot_ids()
        logger.info(f"Retraining {len(bot_ids)} bots: {bot_ids}")
        for bot_id in bot_ids:
            atr_coeff = self._model_repo.get_atr_coeff(bot_id)
            try:
                self.run(bot_id, atr_coeff, days_of_history)
            except Exception as e:
                logger.error(f"[{bot_id}] Failed: {e}")
