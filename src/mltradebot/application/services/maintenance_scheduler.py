from __future__ import annotations

from datetime import datetime


class MaintenanceScheduler:
    """GMO Coin メンテナンス時間 (水曜 14:00-17:00 JST) を判定する。

    元コードの start_all_bots() 内に埋め込まれていた条件分岐を分離する。
    """

    # weekday(): Monday=0, Wednesday=2
    _MAINTENANCE_WEEKDAY = 2
    _MAINTENANCE_START_HOUR = 14
    _MAINTENANCE_END_HOUR = 17
    _PRE_MAINTENANCE_MINUTE = 1  # 14:01 に事前準備を実行

    def is_pre_maintenance(self, dt: datetime) -> bool:
        """メンテナンス直前 (水曜 14:01) かどうか。"""
        return (
            dt.weekday() == self._MAINTENANCE_WEEKDAY
            and dt.hour == self._MAINTENANCE_START_HOUR
            and dt.minute == self._PRE_MAINTENANCE_MINUTE
        )

    def is_in_maintenance(self, dt: datetime) -> bool:
        """メンテナンス中 (水曜 14:00-17:00) かどうか。"""
        return (
            dt.weekday() == self._MAINTENANCE_WEEKDAY
            and self._MAINTENANCE_START_HOUR <= dt.hour <= self._MAINTENANCE_END_HOUR
        )
