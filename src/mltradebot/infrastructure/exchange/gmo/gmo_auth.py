from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Dict


class GMOAuthenticator:
    """GMO Coin API の HMAC-SHA256 認証ヘッダーを生成する。

    元コードでは各メソッドにコピペされていた署名ロジックをここに集約する。
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def make_headers(
        self,
        method: str,
        path: str,
        body: Dict | None = None,
        time_offset_sec: int = 0,
    ) -> Dict[str, str]:
        """認証ヘッダーを生成する。

        Args:
            method: "GET" or "POST"
            path: API パス (例: "/v1/order")
            body: POST リクエストボディ (dict)
            time_offset_sec: ERR-5008 リトライ用の時刻オフセット (秒)
        """
        ts = int(time.mktime(datetime.now().timetuple())) + time_offset_sec
        timestamp = f"{ts}000"  # GMO は ms 単位のタイムスタンプを要求

        body_str = json.dumps(body) if body else ""
        text = timestamp + method + path + body_str

        sign = hmac.new(
            self._secret_key.encode("ascii"),
            text.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "API-KEY": self._api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign,
        }
