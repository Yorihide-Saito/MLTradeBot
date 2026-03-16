from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import List


class LotAllocator:
    """証拠金残高から各ボットのロットサイズを計算して配分する。

    元コードの Manager.update_order_lot_all_bots() と
    round_off_to_two_decimal_places() のロジックを抽出したもの。
    """

    def __init__(self, available_margin_ratio: float) -> None:
        assert 0.0 <= available_margin_ratio < 0.85, "available_margin_ratio must be in [0, 0.85)"
        self._ratio = available_margin_ratio

    def allocate(
        self,
        jpy_balance: int,
        current_price: float,
        n_bots: int,
    ) -> List[float]:
        """各ボットへのロット配分リストを返す (大きい順)。

        Args:
            jpy_balance: JPY 建て口座残高
            current_price: 現在の BTC 価格
            n_bots: ボット数

        Returns:
            ロットのリスト (インデックス 0 が最大)
        """
        max_lot_raw = jpy_balance / current_price * self._ratio
        max_lot = self._round_down_2dp(max_lot_raw)
        max_lot_int = int(max_lot * 100)

        lot_list = [
            (max_lot_int + i) // n_bots / 100
            for i in range(n_bots)
        ]
        return list(reversed(lot_list))

    @staticmethod
    def _round_down_2dp(value: float) -> float:
        two_places = Decimal(10) ** -2
        return float(Decimal(str(value)).quantize(two_places, rounding=ROUND_DOWN))
