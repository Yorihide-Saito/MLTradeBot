"""
tests/unit/test_auth.py — 認証ヘッダー生成の単体テスト
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from freezegun import freeze_time

from src.infrastructure.exchange.bitflyer.bitflyer_auth import BitFlyerAuthenticator
from src.infrastructure.exchange.gmo.gmo_auth import GMOAuthenticator


FIXED_TIMESTAMP = "2024-01-15 12:00:00"


class TestBitFlyerAuthenticator:
    @freeze_time(FIXED_TIMESTAMP)
    def test_headers_contain_required_keys(self):
        auth = BitFlyerAuthenticator("test_key", "test_secret")
        headers = auth.make_headers("GET", "/v1/me/getbalance")
        assert "ACCESS-KEY" in headers
        assert "ACCESS-TIMESTAMP" in headers
        assert "ACCESS-SIGN" in headers
        assert "Content-Type" in headers

    @freeze_time(FIXED_TIMESTAMP)
    def test_api_key_in_header(self):
        auth = BitFlyerAuthenticator("my_api_key", "secret")
        headers = auth.make_headers("GET", "/v1/me/getbalance")
        assert headers["ACCESS-KEY"] == "my_api_key"

    @freeze_time(FIXED_TIMESTAMP)
    def test_timestamp_is_unix_seconds(self):
        """BitFlyer は秒単位のタイムスタンプを要求 (GMO と異なり ms ではない)。"""
        auth = BitFlyerAuthenticator("key", "secret")
        headers = auth.make_headers("GET", "/v1/me/getbalance")
        ts = headers["ACCESS-TIMESTAMP"]
        # 10 桁 = 秒単位
        assert len(ts) == 10

    @freeze_time(FIXED_TIMESTAMP)
    def test_hmac_signature_is_correct(self):
        api_key = "test_key"
        secret = "test_secret"
        method = "POST"
        path = "/v1/me/sendchildorder"
        body = '{"product_code":"FX_BTC_JPY"}'

        auth = BitFlyerAuthenticator(api_key, secret)
        headers = auth.make_headers(method, path, body)

        ts = headers["ACCESS-TIMESTAMP"]
        expected_text = ts + method + path + body
        expected_sign = hmac.new(
            secret.encode("utf-8"),
            expected_text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        assert headers["ACCESS-SIGN"] == expected_sign

    @freeze_time(FIXED_TIMESTAMP)
    def test_get_request_with_no_body(self):
        auth = BitFlyerAuthenticator("key", "secret")
        headers = auth.make_headers("GET", "/v1/ticker")
        # 署名生成がエラーを起こさないこと
        assert headers["ACCESS-SIGN"] != ""


class TestGMOAuthenticator:
    @freeze_time(FIXED_TIMESTAMP)
    def test_headers_contain_required_keys(self):
        auth = GMOAuthenticator("test_key", "test_secret")
        headers = auth.make_headers("GET", "/v1/account/assets")
        assert "API-KEY" in headers
        assert "API-TIMESTAMP" in headers
        assert "API-SIGN" in headers

    @freeze_time(FIXED_TIMESTAMP)
    def test_timestamp_is_milliseconds(self):
        """GMO Coin は ms 単位のタイムスタンプを要求 (末尾 '000')。"""
        auth = GMOAuthenticator("key", "secret")
        headers = auth.make_headers("GET", "/v1/account/assets")
        ts = headers["API-TIMESTAMP"]
        # 13 桁 = ms 単位
        assert len(ts) == 13
        assert ts.endswith("000")

    @freeze_time(FIXED_TIMESTAMP)
    def test_hmac_signature_is_correct_for_post(self):
        api_key = "test_key"
        secret = "test_secret"
        method = "POST"
        path = "/v1/order"
        body = {"symbol": "BTC_JPY", "side": "BUY", "executionType": "LIMIT"}

        auth = GMOAuthenticator(api_key, secret)
        headers = auth.make_headers(method, path, body)

        ts = headers["API-TIMESTAMP"]
        body_str = json.dumps(body)
        expected_text = ts + method + path + body_str
        expected_sign = hmac.new(
            secret.encode("ascii"),
            expected_text.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

        assert headers["API-SIGN"] == expected_sign

    @freeze_time(FIXED_TIMESTAMP)
    def test_time_offset_shifts_timestamp(self):
        auth = GMOAuthenticator("key", "secret")
        headers_no_offset = auth.make_headers("GET", "/v1/order", time_offset_sec=0)
        headers_with_offset = auth.make_headers("GET", "/v1/order", time_offset_sec=10)

        ts_no = int(headers_no_offset["API-TIMESTAMP"]) // 1000
        ts_with = int(headers_with_offset["API-TIMESTAMP"]) // 1000
        assert ts_with - ts_no == 10
