from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class SignalGeneratorPort(ABC):
    """売買シグナル生成の抽象インターフェース。LightGBM等を差し替え可能にする。"""

    @abstractmethod
    def predict(self, df_features: pd.DataFrame, feature_names: List[str]) -> float:
        """
        スコアを返す。正の値がシグナルあり。
        df_features の [-2] 行（直近確定足）を使用すること。
        """
