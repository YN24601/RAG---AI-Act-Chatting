"""Download the EUR-Lex HTML snapshot of the EU AI Act and record provenance.

Idempotent: skips download if the snapshot already exists unless force=True.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date

import requests

from .schema import (
    CELEX,
    FETCH_META_PATH,
    RAW_HTML_PATH,
    SOURCE_URL,
    VERSION,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch(force: bool = False) -> str:
    """Ensure the raw HTML snapshot exists locally; return its text.

    Writes a sidecar fetch_metadata.json with provenance (url, date, sha256).
    """
    RAW_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_HTML_PATH.exists() and not force:
        print(f"[fetch] snapshot exists, skipping download: {RAW_HTML_PATH}")
        return RAW_HTML_PATH.read_text(encoding="utf-8")

    print(f"[fetch] downloading {SOURCE_URL}")
    resp = requests.get(SOURCE_URL, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    html = resp.text

    RAW_HTML_PATH.write_text(html, encoding="utf-8")
    sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()

    meta = {
        "source_url": SOURCE_URL,
        "celex": CELEX,
        "version": VERSION,
        "fetch_date": date.today().isoformat(),
        "http_status": resp.status_code,
        "bytes": len(html.encode("utf-8")),
        "sha256": sha256,
        "digital_omnibus_included": False,
    }
    FETCH_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fetch] saved {meta['bytes']} bytes, sha256={sha256[:12]}…  -> {RAW_HTML_PATH}")
    return html
