from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

from src.domain.entities.position import Position
from src.domain.ports.state_repository_port import OrderStateValue, StateRepositoryPort


class PickleStateRepository(StateRepositoryPort):
    """注文・ポジション状態を Pickle ファイルで永続化する。

    元コードの load_pickle / save_pickle と orderId_dict / positionId_dict の
    散在したロジックをここに集約する。
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Order state                                                          #
    # ------------------------------------------------------------------ #

    def load_order_state(self, bot_id: str) -> Dict[int, OrderStateValue]:
        path = self._order_path(bot_id)
        return self._load(path) or {}

    def save_order_state(self, bot_id: str, state: Dict[int, OrderStateValue]) -> None:
        self._save(state, self._order_path(bot_id))

    # ------------------------------------------------------------------ #
    # Position state                                                       #
    # ------------------------------------------------------------------ #

    def load_position_state(self, bot_id: str) -> Dict[int, Position]:
        path = self._position_path(bot_id)
        return self._load(path) or {}

    def save_position_state(self, bot_id: str, state: Dict[int, Position]) -> None:
        self._save(state, self._position_path(bot_id))

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _order_path(self, bot_id: str) -> Path:
        return self._cache_dir / f"bot_{bot_id}_orders.pkl"

    def _position_path(self, bot_id: str) -> Path:
        return self._cache_dir / f"bot_{bot_id}_positions.pkl"

    @staticmethod
    def _load(path: Path):
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save(obj, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(obj, f)
