from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class ModelRepositoryPort(ABC):
    """学習済みモデルの読み書きの抽象インターフェース。"""

    @abstractmethod
    def load_model_pair(self, bot_id: str) -> Tuple[Any, Any]:
        """(model_buy, model_sell) を返す。"""

    @abstractmethod
    def save_model_pair(self, bot_id: str, model_buy: Any, model_sell: Any) -> None:
        """モデルペアを保存する。"""

    @abstractmethod
    def list_bot_ids(self) -> List[str]:
        """モデルファイルから bot_id 一覧を返す。"""

    @abstractmethod
    def get_atr_coeff(self, bot_id: str) -> float:
        """bot_id からATR係数を解析して返す (例: '0p186' -> 0.186)。"""
