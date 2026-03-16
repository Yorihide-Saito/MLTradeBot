from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import pandas as pd


class FeatureSelector:
    """features_default.pkl から特徴量名リストを読み込み、DataFrame を絞り込む。"""

    def __init__(self, feature_pkl_path: Path) -> None:
        self._path = feature_pkl_path
        self._names: List[str] | None = None

    def load_feature_names(self) -> List[str]:
        if self._names is None:
            with self._path.open("rb") as f:
                self._names = pickle.load(f)
        return self._names

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        names = self.load_feature_names()
        return df[names]
