"""
training/model_trainer.py — Optuna + Walk-forward CV による LightGBM 学習
"""
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
    """Optuna でハイパーパラメータを探索し、Walk-forward CV で評価する。"""

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

    def train(self, df: pd.DataFrame, label_col: str) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """Optuna で最適パラメータを探索し、全データで最終モデルを学習して返す。"""
        df_clean = df.dropna(subset=self._features + [label_col])
        X, y = df_clean[self._features], df_clean[label_col]
        logger.info(f"Training [{label_col}]: samples={len(X)}, pos_rate={y.mean():.4f}")

        best_params = self._optimize(X, y)
        logger.info(f"Best params: {best_params}")
        return self._fit(X, y, best_params), best_params

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
            for tr_idx, val_idx in tscv.split(X):
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                pred = model.predict_proba(X.iloc[val_idx])[:, 1]
                scores.append(self._precision_at_threshold(y.iloc[val_idx], pred))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._rng),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)
        return study.best_params

    def _fit(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss", verbosity=-1,
            boosting_type="gbdt", random_state=self._rng, **params,
        )
        model.fit(X, y)
        return model

    @staticmethod
    def _precision_at_threshold(y_true: pd.Series, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        mask = y_pred >= threshold
        return float(y_true[mask].mean()) if mask.sum() > 0 else 0.0
