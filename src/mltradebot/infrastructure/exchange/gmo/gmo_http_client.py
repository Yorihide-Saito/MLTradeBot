from __future__ import annotations

import json
import time
from typing import Any, Dict

import httpx
from loguru import logger

from mltradebot.infrastructure.exchange.gmo.gmo_retry import GMORetryHandler

PRIVATE_BASE = "https://api.coin.z.com/private"
PUBLIC_BASE = "https://api.coin.z.com/public"
_RATE_LIMIT_SLEEP = 0.2  # GMO レート制限対策


class GMOHttpClient:
    """GMO Coin API への低レベル HTTP アクセスを担うクライアント。

    - プライベート API (認証あり)
    - パブリック API (認証なし)
    の両方を扱う。リトライロジックは GMORetryHandler に委譲する。
    """

    def __init__(self, retry_handler: GMORetryHandler) -> None:
        self._retry = retry_handler
        self._client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_private(self, path: str, body: Dict) -> Dict[str, Any]:
        time.sleep(_RATE_LIMIT_SLEEP)
        url = PRIVATE_BASE + path

        def _post(headers: Dict[str, str]) -> Dict[str, Any]:
            res = self._client.post(url, headers=headers, content=json.dumps(body))
            return res.json()

        return self._retry.post_with_retry(_post, "POST", path, body)

    def get_private(self, path: str, params: Dict | None = None) -> Dict[str, Any]:
        time.sleep(_RATE_LIMIT_SLEEP)
        url = PRIVATE_BASE + path

        def _get(headers: Dict[str, str]) -> Dict[str, Any]:
            res = self._client.get(url, headers=headers, params=params or {})
            return res.json()

        return self._retry.get_with_retry(_get, "GET", path)

    def get_public(self, path: str, params: Dict | None = None) -> Dict[str, Any]:
        url = PUBLIC_BASE + path
        cnt = 0
        while True:
            res = self._client.get(url, params=params or {})
            data = res.json()
            if "data" in data:
                return data
            logger.warning(f"Public API no 'data' key: {data}")
            time.sleep(1.1)
            cnt += 1
            if cnt >= 10:
                raise TimeoutError(f"GMO public API did not return data after 10 retries: {path}")

    def close(self) -> None:
        self._client.close()
