from __future__ import annotations

import json
from typing import Dict, List

from loguru import logger

from mltradebot.domain.entities.order import (
    Execution,
    ExecutionType,
    Order,
    OrderSide,
    OrderStatus,
    SettleType,
)
from mltradebot.domain.entities.position import Position, PositionSummary
from mltradebot.domain.ports.exchange_port import ExchangePort
from mltradebot.infrastructure.exchange.gmo.gmo_http_client import GMOHttpClient


class GMOExchangeAdapter(ExchangePort):
    """GMO Coin Private API を ExchangePort インターフェースで包む Adapter。

    各メソッドは元の gmocoin.py のメソッドに 1:1 対応しているが、
    リトライ・認証・HTTP は下位レイヤーに委譲済みなのでシンプルになっている。
    """

    def __init__(self, http: GMOHttpClient) -> None:
        self._http = http

    # ------------------------------------------------------------------ #
    # Orders                                                               #
    # ------------------------------------------------------------------ #

    def place_limit_order(self, symbol: str, side: OrderSide, size: float, price: int) -> int:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "LIMIT",
            "size": str(round(size, 2)),
            "price": str(price),
        }
        res = self._http.post_private("/v1/order", body)
        order_id = int(res["data"])
        logger.info(f"place_limit_order: {side} {size} @ {price} -> order_id={order_id}")
        return order_id

    def place_market_order(self, symbol: str, side: OrderSide, size: float) -> int:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "MARKET",
            "size": str(round(size, 2)),
        }
        res = self._http.post_private("/v1/order", body)
        return int(res["data"])

    def place_limit_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
        position_id: int,
    ) -> int:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "LIMIT",
            "price": str(price),
            "settlePosition": [{"positionId": position_id, "size": str(round(size, 2))}],
        }
        res = self._http.post_private("/v1/closeOrder", body)
        order_id = int(res["data"])
        logger.info(
            f"place_limit_close_order: {side} {size} @ {price} pos={position_id} -> {order_id}"
        )
        return order_id

    def place_market_close_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        position_id: int,
    ) -> None:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "MARKET",
            "settlePosition": [{"positionId": position_id, "size": str(round(size, 2))}],
        }
        self._http.post_private("/v1/closeOrder", body)
        logger.info(f"place_market_close_order: {side} {size} pos={position_id}")

    def place_limit_close_bulk_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: int,
    ) -> int:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "LIMIT",
            "size": str(round(size, 2)),
            "price": str(price),
        }
        res = self._http.post_private("/v1/closeBulkOrder", body)
        return int(res["data"])

    def place_market_close_bulk_order(self, symbol: str, side: OrderSide, size: float) -> None:
        body = {
            "symbol": symbol,
            "side": side.value,
            "executionType": "MARKET",
            "size": str(round(size, 2)),
        }
        self._http.post_private("/v1/closeBulkOrder", body)
        logger.info(f"place_market_close_bulk_order: {side} {size}")

    # ------------------------------------------------------------------ #
    # Cancellations                                                        #
    # ------------------------------------------------------------------ #

    def cancel_orders(self, order_ids: List[int]) -> None:
        if not order_ids:
            return
        body = {"orderIds": order_ids}
        res = self._http.post_private("/v1/cancelOrders", body)
        if res.get("data", {}).get("success"):
            logger.info(f"cancel_orders: {json.dumps(res, ensure_ascii=False)}")

    def cancel_all_orders(self, symbol: str) -> None:
        body = {"symbols": [symbol]}
        self._http.post_private("/v1/cancelBulkOrder", body)
        logger.info(f"cancel_all_orders: {symbol}")

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_active_orders(self, symbol: str) -> List[Order]:
        res = self._http.get_private("/v1/activeOrders", {"symbol": symbol})
        orders = []
        for o in res.get("data", {}).get("list", []):
            orders.append(
                Order(
                    order_id=int(o["orderId"]),
                    symbol=o["symbol"],
                    side=OrderSide(o["side"]),
                    execution_type=ExecutionType(o["executionType"]),
                    size=float(o["size"]),
                    price=int(o["price"]) if o.get("price") else None,
                    status=OrderStatus.PENDING,
                )
            )
        return orders

    def get_position_summary(self, symbol: str) -> PositionSummary:
        res = self._http.get_private("/v1/positionSummary", {"symbol": symbol})
        summary = PositionSummary()
        for p in res.get("data", {}).get("list", []):
            if p["side"] == "BUY":
                summary.buy_quantity = float(p["sumPositionQuantity"])
                summary.buy_avg_rate = float(p.get("averagePositionRate", 0))
            elif p["side"] == "SELL":
                summary.sell_quantity = float(p["sumPositionQuantity"])
                summary.sell_avg_rate = float(p.get("averagePositionRate", 0))
        return summary

    def get_open_positions(self, symbol: str) -> Dict[int, Position]:
        res = self._http.get_private("/v1/positionSummary", {"symbol": symbol})
        positions: Dict[int, Position] = {}
        for p in res.get("data", {}).get("list", []):
            pos_id = int(p.get("positionId", 0))
            positions[pos_id] = Position(
                position_id=pos_id,
                settle_type=SettleType.OPEN,
                side=OrderSide(p["side"]),
                size=float(p["sumPositionQuantity"]),
                entry_price=int(float(p.get("averagePositionRate", 0))),
            )
        return positions

    def get_account_jpy_balance(self) -> int:
        res = self._http.get_private("/v1/account/assets")
        for asset in res.get("data", []):
            if asset["symbol"] == "JPY":
                return int(asset["amount"])
        return 0

    def get_recent_executions(self, symbol: str, count: int = 100) -> List[Execution]:
        params = {"symbol": symbol, "page": 1, "count": count}
        res = self._http.get_private("/v1/latestExecutions", params)
        executions = []
        for e in res.get("data", {}).get("list", []):
            from datetime import datetime, timezone
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            executions.append(
                Execution(
                    order_id=int(e["orderId"]),
                    position_id=int(e.get("positionId", 0)),
                    settle_type=SettleType(e["settleType"]),
                    side=OrderSide(e["side"]),
                    size=float(e["size"]),
                    price=float(e["price"]),
                    loss_gain=float(e.get("lossGain", 0)),
                    timestamp=ts,
                )
            )
        return executions
