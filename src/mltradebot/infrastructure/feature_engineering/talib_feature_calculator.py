from __future__ import annotations

import warnings

import pandas as pd
import talib

# 元の richman_features.py の calc_features() と完全に同一の計算を維持する。
# 既存の学習済みモデル (.xz) は特定の特徴量セットで学習されているため、
# 特徴量名・計算式を変更すると推論結果が壊れる。


class TALibFeatureCalculator:
    """TA-Lib を使用してテクニカル指標を計算する。

    元コードの richman_features.calc_features() を
    クラスに移植したもの。計算内容は一切変更していない。
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """50+ のテクニカル指標を計算して df に追加する。

        Args:
            df: OHLCVデータ (列: op, hi, lo, cl, volume)

        Returns:
            テクニカル指標を追加した DataFrame
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._calc(df.copy())

    @staticmethod
    def _calc(df: pd.DataFrame) -> pd.DataFrame:
        op = df["op"]
        high = df["hi"]
        low = df["lo"]
        close = df["cl"]
        volume = df["volume"]

        hilo = (df["hi"] + df["lo"]) / 2

        # ---- Overlap Studies ----
        df["BBANDS_upperband"], df["BBANDS_middleband"], df["BBANDS_lowerband"] = talib.BBANDS(
            close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
        )
        df["BBANDS_upperband"] -= hilo
        df["BBANDS_middleband"] -= hilo
        df["BBANDS_lowerband"] -= hilo
        df["DEMA"] = talib.DEMA(close, timeperiod=30) - hilo
        df["EMA"] = talib.EMA(close, timeperiod=30) - hilo
        df["HT_TRENDLINE"] = talib.HT_TRENDLINE(close) - hilo
        df["KAMA"] = talib.KAMA(close, timeperiod=30) - hilo
        df["MA"] = talib.MA(close, timeperiod=30, matype=0) - hilo
        df["MIDPOINT"] = talib.MIDPOINT(close, timeperiod=14) - hilo
        df["SMA"] = talib.SMA(close, timeperiod=30) - hilo
        df["T3"] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
        df["TEMA"] = talib.TEMA(close, timeperiod=30) - hilo
        df["TRIMA"] = talib.TRIMA(close, timeperiod=30) - hilo
        df["WMA"] = talib.WMA(close, timeperiod=30) - hilo

        # ---- Momentum Indicators ----
        df["ADX"] = talib.ADX(high, low, close, timeperiod=14)
        df["ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
        df["APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        df["AROON_aroondown"], df["AROON_aroonup"] = talib.AROON(high, low, timeperiod=14)
        df["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
        df["BOP"] = talib.BOP(op, high, low, close)
        df["CCI"] = talib.CCI(high, low, close, timeperiod=14)
        df["DX"] = talib.DX(high, low, close, timeperiod=14)
        df["MACD_macd"], df["MACD_macdsignal"], df["MACD_macdhist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["MFI"] = talib.MFI(high, low, close, volume, timeperiod=14)
        df["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df["MINUS_DM"] = talib.MINUS_DM(high, low, timeperiod=14)
        df["MOM"] = talib.MOM(close, timeperiod=10)
        df["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df["PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)
        df["RSI"] = talib.RSI(close, timeperiod=14)
        df["STOCH_slowk"], df["STOCH_slowd"] = talib.STOCH(
            high, low, close,
            fastk_period=5, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0,
        )
        df["STOCHF_fastk"], df["STOCHF_fastd"] = talib.STOCHF(
            high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        df["STOCHRSI_fastk"], df["STOCHRSI_fastd"] = talib.STOCHRSI(
            close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        df["TRIX"] = talib.TRIX(close, timeperiod=30)
        df["ULTOSC"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df["WILLR"] = talib.WILLR(high, low, close, timeperiod=14)

        # ---- Volume Indicators ----
        df["AD"] = talib.AD(high, low, close, volume)
        df["ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df["OBV"] = talib.OBV(close, volume)

        # ---- Volatility Indicators ----
        df["ATR"] = talib.ATR(high, low, close, timeperiod=14)
        df["NATR"] = talib.NATR(high, low, close, timeperiod=14)
        df["TRANGE"] = talib.TRANGE(high, low, close)

        # ---- Cycle Indicators ----
        df["HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
        df["HT_DCPHASE"] = talib.HT_DCPHASE(close)
        df["HT_PHASOR_inphase"], df["HT_PHASOR_quadrature"] = talib.HT_PHASOR(close)
        df["HT_SINE_sine"], df["HT_SINE_leadsine"] = talib.HT_SINE(close)
        df["HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

        # ---- Statistic Functions ----
        df["BETA"] = talib.BETA(high, low, timeperiod=5)
        df["CORREL"] = talib.CORREL(high, low, timeperiod=30)
        df["LINEARREG"] = talib.LINEARREG(close, timeperiod=14) - close
        df["LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(close, timeperiod=14)
        df["LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
        df["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(close, timeperiod=14)
        df["STDDEV"] = talib.STDDEV(close, timeperiod=5, nbdev=1)

        return df
