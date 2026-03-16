from __future__ import annotations

from typing import Any, List

import pandas as pd

from src.domain.ports.signal_generator_port import SignalGeneratorPort


class LightGBMSignalGenerator(SignalGeneratorPort):
    """LightGBM モデルを SignalGeneratorPort として包む Adapter。

    元コードの GMOBot.predict_order() の推論部分を抽出したもの。
    [-2] 行 (直近確定足) を使用するルールはここで守る。
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, df_features: pd.DataFrame, feature_names: List[str]) -> float:
        """直近確定足 ([-2] 行) のスコアを返す。正の値がシグナルあり。"""
        sample = df_features[feature_names].iloc[-2:].copy()
        score = float(self._model.predict(sample)[-1])
        return score
