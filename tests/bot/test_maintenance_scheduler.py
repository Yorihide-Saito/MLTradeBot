"""
tests/unit/test_maintenance_scheduler.py — MaintenanceScheduler の単体テスト
"""
from __future__ import annotations

from datetime import datetime

import pytest

from mltradebot.application.services.maintenance_scheduler import MaintenanceScheduler


@pytest.fixture
def scheduler() -> MaintenanceScheduler:
    return MaintenanceScheduler()


class TestIsPreMaintenance:
    def test_wednesday_14_01_is_pre_maintenance(self, scheduler):
        # 2024-01-03 は水曜日
        dt = datetime(2024, 1, 3, 14, 1)
        assert scheduler.is_pre_maintenance(dt) is True

    def test_wednesday_14_00_is_not_pre_maintenance(self, scheduler):
        dt = datetime(2024, 1, 3, 14, 0)
        assert scheduler.is_pre_maintenance(dt) is False

    def test_wednesday_14_02_is_not_pre_maintenance(self, scheduler):
        dt = datetime(2024, 1, 3, 14, 2)
        assert scheduler.is_pre_maintenance(dt) is False

    def test_tuesday_14_01_is_not_pre_maintenance(self, scheduler):
        # 2024-01-02 は火曜日
        dt = datetime(2024, 1, 2, 14, 1)
        assert scheduler.is_pre_maintenance(dt) is False

    def test_thursday_14_01_is_not_pre_maintenance(self, scheduler):
        # 2024-01-04 は木曜日
        dt = datetime(2024, 1, 4, 14, 1)
        assert scheduler.is_pre_maintenance(dt) is False


class TestIsInMaintenance:
    @pytest.mark.parametrize("hour", [14, 15, 16, 17])
    def test_wednesday_maintenance_hours(self, scheduler, hour):
        dt = datetime(2024, 1, 3, hour, 0)
        assert scheduler.is_in_maintenance(dt) is True

    def test_wednesday_13_59_is_not_maintenance(self, scheduler):
        dt = datetime(2024, 1, 3, 13, 59)
        assert scheduler.is_in_maintenance(dt) is False

    def test_wednesday_18_00_is_not_maintenance(self, scheduler):
        dt = datetime(2024, 1, 3, 18, 0)
        assert scheduler.is_in_maintenance(dt) is False

    def test_tuesday_15_00_is_not_maintenance(self, scheduler):
        dt = datetime(2024, 1, 2, 15, 0)
        assert scheduler.is_in_maintenance(dt) is False
