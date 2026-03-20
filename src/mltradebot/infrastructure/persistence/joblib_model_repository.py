from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, List, Tuple

import joblib

from mltradebot.domain.ports.model_repository_port import ModelRepositoryPort


class JoblibModelRepository(ModelRepositoryPort):
    """LightGBM モデルを joblib (.xz) 形式で読み書きする。

    ファイル命名規則: buy_{bot_id}.xz / sell_{bot_id}.xz
    bot_id 例: "0p1866346763152701" -> ATR係数 0.1866...
    """

    def __init__(self, model_buy_dir: Path, model_sell_dir: Path) -> None:
        self._buy_dir = model_buy_dir
        self._sell_dir = model_sell_dir

    def load_model_pair(self, bot_id: str) -> Tuple[Any, Any]:
        buy_path = self._find_model(self._buy_dir, bot_id)
        sell_path = self._find_model(self._sell_dir, bot_id)
        return joblib.load(buy_path), joblib.load(sell_path)

    def save_model_pair(self, bot_id: str, model_buy: Any, model_sell: Any) -> None:
        self._buy_dir.mkdir(parents=True, exist_ok=True)
        self._sell_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_buy, self._buy_dir / f"buy_{bot_id}.xz")
        joblib.dump(model_sell, self._sell_dir / f"sell_{bot_id}.xz")

    def list_bot_ids(self) -> List[str]:
        """model_buy ディレクトリのファイルから bot_id を抽出する。"""
        pattern = str(self._buy_dir / "**")
        paths = sorted(
            p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)
        )
        return [self._extract_bot_id(p) for p in paths]

    def get_atr_coeff(self, bot_id: str) -> float:
        # "0p1866346763152701" -> "0.1866346763152701" -> float
        return float(bot_id.replace("p", "."))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_bot_id(filepath: str) -> str:
        # "/path/to/buy_0p1866.xz" -> "0p1866"
        basename = os.path.basename(filepath)
        return basename.split("_")[-1].split(".")[0]

    def _find_model(self, directory: Path, bot_id: str) -> Path:
        pattern = str(directory / f"*_{bot_id}.*")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No model file found for bot_id={bot_id} in {directory}")
        return Path(sorted(matches)[0])
