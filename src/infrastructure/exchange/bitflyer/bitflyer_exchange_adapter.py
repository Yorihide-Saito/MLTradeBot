from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

from loguru import logger

from src.domain.entities.order import (
    Execution,
    ExecutionType,
    Order,
    OrderSide,
    OrderStatus,
    SettleType,
)
from src.domain.entities.position import Position, PositionSummary
from src.domain.ports.exchange_port import ExchangePort
from src.infrastructure.exchange.bitflyer.bitflyer_http_client import BitFlyerHttpClient


class BitFlyerExchangeAdapter(ExchangePort):
    """BitFlyer Lightning REST API を ExchangePort インターフェースで包む Adapter。

    対象: FX_BTC_JPY (証拠金取引)

    GMO Coin との主な違い:
    - 注文返り値: child_order_acceptance_id (文字列) を int にハッシュして管理
    - ポジションクローズ: GMO の settlePosition に相当する概念がなく、
      反対方向の注文でネットアウトするか、get_positions で個別 positionId を管理
    - cancel 時は child_order_acceptance_id を使用
    - cancel_all は POST /v1/me/cancelallchildorders
    """

    PRODUCT_CODE = "FX_BTC_JPY"

    def __init__(self, http: BitFlyerHttpClient) -> None:
        self._http = http
        # acceptance_id (str) <-> 疑似 order_id (int) の対応表
        self._id_map: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Orders                                                               #
    # ------------------------------------------------------------------ #

    def place_limit_order(self, symbol: str, side: OrderSide, size: float, price: int) -> int:
        body = {
            "product_code": self.PRODUCT_CODE,
            "child_order_type": "LIMIT",
            "side": side.value,
            "price": price,
            "size": round(size, 2),
            "minute_to_expire": 10000,
            "time_in_force": "GTC",
        }
        res = self._http.post_private("/v1/me/sendchildorder", body)
        order_id = self._acceptance_to_int(res["child_order_acceptance_id"])
        self._id_map[order_id] = res["child_order_acceptance_id"]
        logger.info(f"BF place_limit_order: {side} {size} @ {price} -> {order_id}")
        return order_id

    def place_market_order(self, symbol: str, side: OrderSide, size: float) -> int:
        body = {
            "product_code": self.PRODUCT_CODE,
            "child_order_type": "MARKET",
            "side": side.value,
            "size": round(size, 2),
            "minute_to_expire": 10000,
            "time_in_force": "GTC",
        }
        res = self._http.post_private("/v1/me/sendchildorder", body)
        order_id = self._acceptance_to_int(res["child_order_acceptance_id"])
        self._id_map[order_id] = res["child_order_acceptance_id"]
        return order_id

    def place_limit_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
        position_id: int,
    ) -> int:
        # BitFlyer FX はポジション指定クローズがないため、
        # 反対方向の指値注文でネットアウトする
        return self.place_limit_order(symbol, side, size, price)

    def place_market_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        position_id: int,
    ) -> None:
        self.place_market_order(symbol, side, size)

    def place_limit_close_bulk_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
    ) -> int:
        return self.place_limit_order(symbol, side, size, price)

    def place_market_close_bulk_order(self, symbol: str, side: OrderSide, size: float) -> None:
        self.place_market_order(symbol, side, size)

    # ------------------------------------------------------------------ #
    # Cancellations                                                        #
    # ------------------------------------------------------------------ #

    def cancel_orders(self, order_ids: List[int]) -> None:
        for oid in order_ids:
            acceptance_id = self._id_map.get(oid)
            if not acceptance_id:
                logger.warning(f"BF cancel_orders: no acceptance_id for order_id={oid}")
                continue
            body = {
                "product_code": self.PRODUCT_CODE,
                "child_order_acceptance_id": acceptance_id,
            }
            self._http.post_private("/v1/me/cancelchildorder", body)
            self._id_map.pop(oid, None)

    def cancel_all_orders(self, symbol: str) -> None:
        body = {"product_code": self.PRODUCT_CODE}
        self._http.post_private("/v1/me/cancelallchildorders", body)
        self._id_map.clear()
        logger.info(f"BF cancel_all_orders: {self.PRODUCT_CODE}")

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_active_orders(self, symbol: str) -> List[Order]:
        params = {
            "product_code": self.PRODUCT_CODE,
            "child_order_state": "ACTIVE",
            "count": 100,
        }
        orders_data = self._http.get_private("/v1/me/getchildorders", params)
        orders = []
        for o in orders_data:
            acceptance_id = o["child_order_acceptance_id"]
            oid = self._acceptance_to_int(acceptance_id)
            self._id_map[oid] = acceptance_id
            orders.append(
                Order(
                    order_id=oid,
                    symbol=self.PRODUCT_CODE,
                    side=OrderSide(o["side"]),
                    execution_type=ExecutionType(o["child_order_type"]),
                    size=float(o["size"]),
                    price=int(o["price"]) if o.get("price") else None,
                    status=OrderStatus.PENDING,
                )
            )
        return orders

    def get_position_summary(self, symbol: str) -> PositionSummary:
        positions = self._http.get_private(
            "/v1/me/getpositions", {"product_code": self.PRODUCT_CODE}
        )
        summary = PositionSummary()
        total_buy = 0.0
        total_sell = 0.0
        buy_value = 0.0
        sell_value = 0.0
        for p in positions:
            size = float(p["size"])
            price = float(p["open_price"])
            if p["side"] == "BUY":
                total_buy += size
                buy_value += size * price
            else:
                total_sell += size
                sell_value += size * price
        summary.buy_quantity = total_buy
        summary.sell_quantity = total_sell
        summary.buy_avg_rate = buy_value / total_buy if total_buy > 0 else 0.0
        summary.sell_avg_rate = sell_value / total_sell if total_sell > 0 else 0.0
        return summary

    def get_open_positions(self, symbol: str) -> Dict[int, Position]:
        positions = self._http.get_private(
            "/v1/me/getpositions", {"product_code": self.PRODUCT_CODE}
        )
        result: Dict[int, Position] = {}
        for p in positions:
            # BitFlyer のポジションIDは文字列; int にハッシュして管理
            pos_id = self._acceptance_to_int(str(p.get("id", p.get("open_price", 0))))
            result[pos_id] = Position(
                position_id=pos_id,
                settle_type=SettleType.OPEN,
                side=OrderSide(p["side"]),
                size=float(p["size"]),
                entry_price=int(float(p["open_price"])),
            )
        return result

    def get_account_jpy_balance(self) -> int:
        balances = self._http.get_private("/v1/me/getbalance")
        for b in balances:
            if b["currency_code"] == "JPY":
                return int(b["amount"])
        return 0

    def get_recent_executions(self, symbol: str, count: int = 100) -> List[Execution]:
        params = {"product_code": self.PRODUCT_CODE, "count": count}
        execs = self._http.get_private("/v1/me/getexecutions", params)
        result = []
        for e in execs:
            ts = datetime.fromisoformat(e["exec_date"].replace("Z", "+00:00"))
            # BitFlyer の約定はポジション概念がやや異なるため、
            # settle_type を side で推定 (open はすべて OPEN とみなす)
            result.append(
                Execution(
                    order_id=self._acceptance_to_int(
                        str(e.get("child_order_acceptance_id", e["id"]))
                    ),
                    position_id=int(e.get("id", 0)),
                    settle_type=SettleType.OPEN,
                    side=OrderSide(e["side"]),
                    size=float(e["size"]),
                    price=float(e["price"]),
                    loss_gain=0.0,  # BitFlyer は約定時に P&L を返さない
                    timestamp=ts,
                )
            )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _acceptance_to_int(acceptance_id: str) -> int:
        """acceptance_id 文字列を一意な int に変換する。"""
        return abs(hash(acceptance_id)) % (10**15)
