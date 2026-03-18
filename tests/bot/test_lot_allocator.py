"""
tests/unit/test_lot_allocator.py — LotAllocator の単体テスト
"""
from __future__ import annotations

import pytest

from src.application.services.lot_allocator import LotAllocator


class TestLotAllocator:
    def test_single_bot_basic(self):
        allocator = LotAllocator(available_margin_ratio=0.5)
        lots = allocator.allocate(jpy_balance=10_000_000, current_price=5_000_000, n_bots=1)
        assert len(lots) == 1
        # 10_000_000 / 5_000_000 * 0.5 = 1.00
        assert lots[0] == pytest.approx(1.00)

    def test_multiple_bots_descending_order(self):
        allocator = LotAllocator(available_margin_ratio=0.5)
        lots = allocator.allocate(jpy_balance=10_000_000, current_price=5_000_000, n_bots=3)
        assert len(lots) == 3
        # リストは降順
        assert lots[0] >= lots[1] >= lots[2]

    def test_rounding_down_to_3dp(self):
        allocator = LotAllocator(available_margin_ratio=0.5)
        # 余剰が生じる割り切れないケース
        lots = allocator.allocate(jpy_balance=1_234_567, current_price=5_000_000, n_bots=2)
        for lot in lots:
            # 小数点以下 3 桁まで (スポット取引の 0.001 BTC 単位対応)
            assert round(lot, 3) == lot

    def test_ratio_boundary_assertion(self):
        with pytest.raises(AssertionError):
            LotAllocator(available_margin_ratio=0.85)

    def test_ratio_zero_allowed(self):
        allocator = LotAllocator(available_margin_ratio=0.0)
        lots = allocator.allocate(jpy_balance=10_000_000, current_price=5_000_000, n_bots=2)
        assert all(lot == 0.0 for lot in lots)

    def test_lot_sum_does_not_exceed_max(self):
        """全ボットのロット合計が max_lot を超えないこと。"""
        allocator = LotAllocator(available_margin_ratio=0.5)
        n = 5
        lots = allocator.allocate(jpy_balance=10_000_000, current_price=5_000_000, n_bots=n)
        max_lot_raw = 10_000_000 / 5_000_000 * 0.5
        assert sum(lots) <= max_lot_raw + 0.01  # 丸め誤差許容
