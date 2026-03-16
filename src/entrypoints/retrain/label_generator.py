from __future__ import annotations

import pandas as pd
import talib


class LabelGenerator:
    """ATR ベースのフォワードリターンラベルを生成する。

    ラベル定義:
    - 買いラベル=1: N本後の終値が現在の終値 + ATR * coeff 以上になった場合
    - 売りラベル=1: N本後の終値が現在の終値 - ATR * coeff 以下になった場合

    元のモデルと同じラベル定義を維持しつつ、新しいデータで再学習する。
    """

    def __init__(
        self,
        atr_coeff: float,
        atr_period: int = 14,
        forward_bars: int = 4,  # 15m * 4 = 1時間先を予測
    ) -> None:
        self._coeff = atr_coeff
        self._atr_period = atr_period
        self._forward_bars = forward_bars

    def generate_buy_labels(self, df: pd.DataFrame) -> pd.Series:
        """買いシグナルラベル (0/1) を生成する。"""
        atr = talib.ATR(df["hi"].values, df["lo"].values, df["cl"].values, timeperiod=self._atr_period)
        threshold = pd.Series(atr, index=df.index) * self._coeff

        future_close = df["cl"].shift(-self._forward_bars)
        label = (future_close >= df["cl"] + threshold).astype(int)
        return label.rename("y_buy")

    def generate_sell_labels(self, df: pd.DataFrame) -> pd.Series:
        """売りシグナルラベル (0/1) を生成する。"""
        atr = talib.ATR(df["hi"].values, df["lo"].values, df["cl"].values, timeperiod=self._atr_period)
        threshold = pd.Series(atr, index=df.index) * self._coeff

        future_close = df["cl"].shift(-self._forward_bars)
        label = (future_close <= df["cl"] - threshold).astype(int)
        return label.rename("y_sell")

    def generate_both(self, df: pd.DataFrame) -> pd.DataFrame:
        """buy / sell 両方のラベルを df に追加して返す。"""
        result = df.copy()
        result["y_buy"] = self.generate_buy_labels(df)
        result["y_sell"] = self.generate_sell_labels(df)
        # 未来が参照できない末尾を除外
        result = result.iloc[: -self._forward_bars].dropna()
        return result
