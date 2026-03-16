from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """環境変数 / .env ファイルから設定を読み込む。

    元コードの GMOBotConfig クラス属性を pydantic-settings に置き換えたもの。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Exchange ----
    exchange_type: Literal["gmo", "bitflyer"] = Field(default="gmo", alias="EXCHANGE_TYPE")

    # GMO Coin
    gmo_api_key: str = Field(default="", alias="GMO_API_KEY")
    gmo_secret_key: str = Field(default="", alias="GMO_SECRET_KEY")

    # BitFlyer
    bitflyer_api_key: str = Field(default="", alias="BITFLYER_API_KEY")
    bitflyer_secret_key: str = Field(default="", alias="BITFLYER_SECRET_KEY")

    # ---- Trading ----
    symbol: str = Field(default="BTC_JPY", alias="SYMBOL")
    pips: int = Field(default=1, alias="PIPS")
    available_margin: float = Field(default=0.5, alias="AVAILABLE_MARGIN")
    atr_period: int = Field(default=14, alias="ATR_PERIOD")
    candle_interval_minutes: int = Field(default=15, alias="CANDLE_INTERVAL_MINUTES")

    # ---- Paths ----
    work_dir: Path = Field(default=Path("/work"), alias="WORK_DIR")

    # ---- Misc ----
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("available_margin")
    @classmethod
    def check_margin(cls, v: float) -> float:
        if not (0.0 <= v < 0.85):
            raise ValueError("available_margin must be in [0.0, 0.85)")
        return v

    # ---- Derived paths ----

    @property
    def model_buy_dir(self) -> Path:
        return self.work_dir / "model_buy"

    @property
    def model_sell_dir(self) -> Path:
        return self.work_dir / "model_sell"

    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def feature_pkl_path(self) -> Path:
        return self.work_dir / "features" / "features_default.pkl"
