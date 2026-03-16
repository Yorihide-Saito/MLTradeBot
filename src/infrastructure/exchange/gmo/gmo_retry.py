from __future__ import annotations

import time
from typing import Any, Callable, Dict

from loguru import logger

from src.infrastructure.exchange.gmo.gmo_auth import GMOAuthenticator

# ERR-5008: タイムスタンプが未来すぎる → 時刻を +td してリトライ
# ERR-5009: タイムスタンプが過去すぎる → 時刻をリセットしてリトライ
_ERR_FUTURE = "ERR-5008"
_ERR_PAST = "ERR-5009"


class GMORetryHandler:
    """ERR-5008 / ERR-5009 リトライロジックを一か所に集約する。

    元コードでは7つのメソッドに同一の40行ブロックがコピペされていた。
    このクラスがその単一の正規実装となる。
    """

    def __init__(self, authenticator: GMOAuthenticator, time_delta_sec: int = 5) -> None:
        self._auth = authenticator
        self._td = time_delta_sec

    def post_with_retry(
        self,
        post_fn: Callable[[Dict[str, str]], Dict[str, Any]],
        method: str,
        path: str,
        body: Dict,
    ) -> Dict[str, Any]:
        """POST リクエストを実行し、タイムスタンプエラー時にリトライする。

        Args:
            post_fn: headers を受け取り dict を返す呼び出し可能オブジェクト
            method: "POST"
            path: API パス
            body: リクエストボディ

        Returns:
            レスポンスの dict
        """
        headers = self._auth.make_headers(method, path, body)
        result = post_fn(headers)

        # ERR-5008: サーバー時刻より未来 → オフセットを加えてリトライ
        while self._is_error(result, _ERR_FUTURE):
            logger.warning(f"ERR-5008 detected, retrying with +{self._td}s offset")
            headers = self._auth.make_headers(method, path, body, time_offset_sec=self._td)
            result = post_fn(headers)
            time.sleep(0.2)

            # ERR-5009: オフセットが大きすぎた → 時刻をリセット
            if self._is_error(result, _ERR_PAST):
                logger.warning("ERR-5009 detected, resetting timestamp")
                headers = self._auth.make_headers(method, path, body)
                result = post_fn(headers)
                time.sleep(0.2)

        return result

    def get_with_retry(
        self,
        get_fn: Callable[[Dict[str, str]], Dict[str, Any]],
        method: str,
        path: str,
    ) -> Dict[str, Any]:
        """GET リクエストを実行する (現状リトライ不要だが統一性のため用意)。"""
        headers = self._auth.make_headers(method, path)
        return get_fn(headers)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_error(result: Dict[str, Any], code: str) -> bool:
        messages = result.get("messages", [])
        return (
            result.get("status") == 1
            and bool(messages)
            and messages[0].get("message_code") == code
        )
