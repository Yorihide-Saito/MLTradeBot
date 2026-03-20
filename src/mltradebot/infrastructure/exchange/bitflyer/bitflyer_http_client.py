from __future__ import annotations

import json
import time
from typing import Any, Dict

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from mltradebot.infrastructure.exchange.bitflyer.bitflyer_auth import BitFlyerAuthenticator

BASE_URL = "https://api.bitflyer.com"
_RATE_LIMIT_SLEEP = 0.2


class BitFlyerHttpClient:
    """BitFlyer Lightning REST API への HTTP アクセスクライアント。"""

    def __init__(self, auth: BitFlyerAuthenticator) -> None:
        self._auth = auth
        self._client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.0))
    def post_private(self, path: str, body: Dict) -> Dict[str, Any]:
        time.sleep(_RATE_LIMIT_SLEEP)
        body_str = json.dumps(body)
        headers = self._auth.make_headers("POST", path, body_str)
        res = self._client.post(BASE_URL + path, headers=headers, content=body_str)
        res.raise_for_status()
        return res.json()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.0))
    def get_private(self, path: str, params: Dict | None = None) -> Any:
        time.sleep(_RATE_LIMIT_SLEEP)
        headers = self._auth.make_headers("GET", path)
        res = self._client.get(BASE_URL + path, headers=headers, params=params or {})
        res.raise_for_status()
        return res.json()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.0))
    def get_public(self, path: str, params: Dict | None = None) -> Any:
        res = self._client.get(BASE_URL + path, params=params or {})
        res.raise_for_status()
        return res.json()

    def close(self) -> None:
        self._client.close()
