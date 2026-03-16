from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, precision_score, recall_score


class ModelEvaluator:
    """学習済みモデルの性能を評価し、メトリクスをレポートする。"""

    def evaluate(
        self,
        model: Any,
        df: pd.DataFrame,
        label_col: str,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """精度・再現率・プロフィットファクター等を計算する。

        Args:
            model: 学習済み LGBMClassifier
            df: 特徴量 + ラベル + OHLCV を含む DataFrame
            label_col: ラベル列名
            feature_names: 特徴量名リスト

        Returns:
            メトリクス辞書
        """
        df_clean = df.dropna(subset=feature_names + [label_col])
        X = df_clean[feature_names]
        y = df_clean[label_col]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)

        # シグナルが出た時の仮想リターン計算
        if "cl" in df_clean.columns:
            returns = df_clean["cl"].pct_change().shift(-1)
            signal_returns = returns[y_pred == 1] if label_col == "y_buy" else -returns[y_pred == 1]
            profit_factor = self._profit_factor(signal_returns)
            sharpe = self._sharpe_ratio(signal_returns)
        else:
            profit_factor = float("nan")
            sharpe = float("nan")

        metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "signal_rate": round(float(y_pred.mean()), 4),
            "profit_factor": round(profit_factor, 4) if not np.isnan(profit_factor) else 0.0,
            "sharpe_ratio": round(sharpe, 4) if not np.isnan(sharpe) else 0.0,
            "positive_label_rate": round(float(y.mean()), 4),
        }

        logger.info(f"[{label_col}] Evaluation: {metrics}")
        return metrics

    def save_report(self, metrics: Dict[str, float], output_dir: Path, run_id: str) -> None:
        """評価レポートを JSON として保存する。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        path = output_dir / f"eval_{run_id}.json"
        with path.open("w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")

    @staticmethod
    def _profit_factor(returns: pd.Series) -> float:
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return float(gains / losses) if losses > 0 else float("nan")

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 365 * 24 * 4) -> float:
        if returns.std() == 0:
            return float("nan")
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
