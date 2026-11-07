"""A tiny client for the Microburbs Hackathon API. Copy it into your project.

    from microburbs_client import Microburbs

    mb = Microburbs()  # reads MICROBURBS_API_KEY (and optional MICROBURBS_API_BASE)
    mb.get("/v1/suburbs/Glebe (NSW)/market/median-rent")
    # -> {"area_name": "Glebe (NSW)", "house": ..., "unit": ..., "as_of": ...}

What it does for you:
- sends your key as a Bearer token;
- URL-encodes the path, so suburb names like "Glebe (NSW)" can be written as-is;
- unwraps the {"data": ...} envelope: you get the data, or None when the API has
  no data for that place (a normal 200 response with "data": null);
- waits and retries on 429 (rate limited) and on brief 5xx hiccups: it honours
  a Retry-After header when the server sends one, otherwise backs off 1, 2, 4... seconds;
- raises MicroburbsError with the API's own message for anything else
  (for example 403 when you ask for a place outside the hackathon area).

Only dependency: requests.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any
from urllib.parse import quote

import requests

DEFAULT_BASE_URL = "https://api-ht.microburbs.com.au"


class MicroburbsError(Exception):
    """An API error that retrying will not fix (bad key, out of area, bad request...)."""

    def __init__(self, status: int, message: str, path: str):
        super().__init__(f"HTTP {status} for {path}: {message}")
        self.status = status
        self.message = message
        self.path = path


class Microburbs:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        timeout: float = 30,
    ):
        self.api_key = api_key or os.environ.get("MICROBURBS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set the MICROBURBS_API_KEY environment variable "
                "(or put it in a .env file) and try again."
            )
        self.base_url = (
            base_url or os.environ.get("MICROBURBS_API_BASE") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        # One Session reuses connections, which is much faster for many calls.
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def __repr__(self) -> str:
        # Never show the key, even by accident in a notebook.
        return f"Microburbs(base_url={self.base_url!r})"

    def get(self, path: str, **params: Any) -> Any:
        """GET an endpoint and return its "data" (None if the API has no data).

        `path` is the endpoint path as written in the docs, e.g.
        "/v1/properties/GANSW705970502/sale-history". Query parameters go in
        as keyword arguments: mb.get("/v1/geocode/suburb", q="surry").
        """
        url = self.base_url + quote(path, safe="/")
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(url, params=params, timeout=self.timeout)

            retryable = resp.status_code == 429 or resp.status_code >= 500
            if retryable and attempt < self.max_retries:
                time.sleep(self._wait_seconds(resp, attempt))
                continue

            if resp.status_code != 200:
                raise MicroburbsError(resp.status_code, self._error_message(resp), path)

            body = resp.json()
            # Success is {"data": ...}; "no data here" is {"data": null, "available": false, ...}.
            return body.get("data") if isinstance(body, dict) and "data" in body else body

        raise AssertionError("unreachable")

    @staticmethod
    def _wait_seconds(resp: requests.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.5)
            except ValueError:
                pass  # Retry-After can also be an HTTP date; fall back to backoff.
        # 1, 2, 4, 8... seconds, plus a little jitter so parallel threads don't retry in lockstep.
        return min(2**attempt, 30) + random.uniform(0, 0.5)

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:300]
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or body)
        return str(body)
