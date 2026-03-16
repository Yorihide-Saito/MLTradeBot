from __future__ import annotations

import hashlib
import hmac
import time
from typing import Dict


class BitFlyerAuthenticator:
    """BitFlyer Lightning API の HMAC-SHA256 認証ヘッダーを生成する。

    GMO Coin との違い:
    - ヘッダー名: ACCESS-KEY / ACCESS-TIMESTAMP / ACCESS-SIGN
    - タイムスタンプ: Unix 秒 (GMO は ms 単位)
    - 署名文字列: timestamp + METHOD + path + body
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key

    def make_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        text = timestamp + method.upper() + path + body

        sign = hmac.new(
            self._secret_key.encode("utf-8"),
            text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "ACCESS-KEY": self._api_key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": sign,
            "Content-Type": "application/json",
        }
