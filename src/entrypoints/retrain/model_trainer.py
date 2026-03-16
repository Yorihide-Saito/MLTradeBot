from __future__ import annotations

from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Walk-forward CV + Optuna で LightGBM モデルをハイパーパラメータ最適化して学習する。

    元コードには再学習パイプラインがなかったため、新規実装。
    時系列データの特性上、通常の k-fold は使わず TimeSeriesSplit を使用する。
    """

    def __init__(
        self,
        feature_names: List[str],
        n_splits: int = 5,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> None:
        self._features = feature_names
        self._n_splits = n_splits
        self._n_trials = n_trials
        self._rng = random_state

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self, df: pd.DataFrame, label_col: str
    ) -> Tuple[lgb.Booster, Dict[str, float]]:
        """Optuna でハイパーパラメータを探索し、最良モデルを全データで学習して返す。

        Args:
            df: 特徴量 + ラベル列を含む DataFrame
            label_col: ラベル列名 ("y_buy" or "y_sell")

        Returns:
            (trained_model, best_params)
        """
        df_clean = df.dropna(subset=self._features + [label_col])
        X = df_clean[self._features]
        y = df_clean[label_col]

        logger.info(f"Training {label_col}: samples={len(X)}, pos_rate={y.mean():.4f}")

        # Optuna で最適パラメータ探索
        best_params = self._optimize(X, y)
        logger.info(f"Best params: {best_params}")

        # 全データで最終モデルを学習
        model = self._fit(X, y, best_params)
        return model, best_params

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _optimize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        tscv = TimeSeriesSplit(n_splits=self._n_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self._rng,
            }

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                pred = model.predict_proba(X_val)[:, 1]
                # precision at top-k として評価 (実際に使われるのは予測スコア > 0)
                scores.append(self._precision_at_threshold(y_val, pred, threshold=0.5))

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._rng),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)
        return study.best_params

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]
    ) -> lgb.LGBMClassifier:
        lgbm_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self._rng,
            **params,
        }
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X, y)
        return model

    @staticmethod
    def _precision_at_threshold(
        y_true: pd.Series, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        mask = y_pred >= threshold
        if mask.sum() == 0:
            return 0.0
        return float(y_true[mask].mean())
