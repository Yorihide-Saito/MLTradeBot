"""
training/label_generator.py — ATR ベースのラベル生成

ラベル定義:
  買いラベル = 1: N本後の終値が 現在終値 + ATR * coeff 以上
  売りラベル = 1: N本後の終値が 現在終値 - ATR * coeff 以下
"""
from __future__ import annotations

import pandas as pd
import talib


class LabelGenerator:
    def __init__(
        self,
        atr_coeff: float,
        atr_period: int = 14,
        forward_bars: int = 4,  # 15m × 4 = 1時間先を予測
    ) -> None:
        self._coeff = atr_coeff
        self._atr_period = atr_period
        self._forward_bars = forward_bars

    def generate_buy_labels(self, df: pd.DataFrame) -> pd.Series:
        """買いシグナルラベル (0/1)。"""
        atr = talib.ATR(
            df["hi"].values, df["lo"].values, df["cl"].values,
            timeperiod=self._atr_period,
        )
        threshold = pd.Series(atr, index=df.index) * self._coeff
        future_close = df["cl"].shift(-self._forward_bars)
        return (future_close >= df["cl"] + threshold).astype(int).rename("y_buy")

    def generate_sell_labels(self, df: pd.DataFrame) -> pd.Series:
        """売りシグナルラベル (0/1)。"""
        atr = talib.ATR(
            df["hi"].values, df["lo"].values, df["cl"].values,
            timeperiod=self._atr_period,
        )
        threshold = pd.Series(atr, index=df.index) * self._coeff
        future_close = df["cl"].shift(-self._forward_bars)
        return (future_close <= df["cl"] - threshold).astype(int).rename("y_sell")

    def generate_both(self, df: pd.DataFrame) -> pd.DataFrame:
        """buy / sell 両方のラベルを df に追加して返す。末尾の forward_bars 行を除外。"""
        result = df.copy()
        result["y_buy"] = self.generate_buy_labels(df)
        result["y_sell"] = self.generate_sell_labels(df)
        return result.iloc[: -self._forward_bars].dropna()
