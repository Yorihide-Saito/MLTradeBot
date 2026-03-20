"""
tests/unit/test_label_generator.py — LabelGenerator の単体テスト
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mltradebot.training.label_generator import LabelGenerator


@pytest.fixture
def df_ohlcv() -> pd.DataFrame:
    """単調増加する OHLCV。買いラベルが出やすい。"""
    n = 50
    close = np.linspace(5_000_000, 5_500_000, n)
    high = close + 50_000
    low = close - 50_000
    return pd.DataFrame(
        {"op": close, "hi": high, "lo": low, "cl": close, "volume": np.ones(n)},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )


class TestLabelGenerator:
    def test_buy_labels_are_binary(self, df_ohlcv):
        gen = LabelGenerator(atr_coeff=0.186)
        labels = gen.generate_buy_labels(df_ohlcv)
        assert set(labels.unique()).issubset({0, 1})

    def test_sell_labels_are_binary(self, df_ohlcv):
        gen = LabelGenerator(atr_coeff=0.186)
        labels = gen.generate_sell_labels(df_ohlcv)
        assert set(labels.unique()).issubset({0, 1})

    def test_generate_both_adds_columns(self, df_ohlcv):
        gen = LabelGenerator(atr_coeff=0.186, forward_bars=4)
        result = gen.generate_both(df_ohlcv)
        assert "y_buy" in result.columns
        assert "y_sell" in result.columns

    def test_generate_both_trims_trailing_rows(self, df_ohlcv):
        forward_bars = 4
        gen = LabelGenerator(atr_coeff=0.186, forward_bars=forward_bars)
        result = gen.generate_both(df_ohlcv)
        # 末尾 forward_bars 行が除去されている
        assert len(result) <= len(df_ohlcv) - forward_bars

    def test_uptrend_produces_buy_labels(self, df_ohlcv):
        """単調増加データでは買いラベルが 1 になりやすい。"""
        gen = LabelGenerator(atr_coeff=0.01, forward_bars=4)
        result = gen.generate_both(df_ohlcv)
        # 上昇トレンドなので買いラベルの正例が存在するはず
        assert result["y_buy"].sum() > 0

    def test_no_nan_in_labels_after_generate_both(self, df_ohlcv):
        gen = LabelGenerator(atr_coeff=0.186, forward_bars=4)
        result = gen.generate_both(df_ohlcv)
        assert result["y_buy"].isna().sum() == 0
        assert result["y_sell"].isna().sum() == 0

    def test_coeff_affects_label_rate(self, df_ohlcv):
        """ATR coeff が大きいほど正例が減る。"""
        gen_small = LabelGenerator(atr_coeff=0.01, forward_bars=4)
        gen_large = LabelGenerator(atr_coeff=5.0, forward_bars=4)
        result_small = gen_small.generate_both(df_ohlcv)
        result_large = gen_large.generate_both(df_ohlcv)
        assert result_small["y_buy"].mean() >= result_large["y_buy"].mean()
