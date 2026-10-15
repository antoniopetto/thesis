"""Download HTTP con cache su disco, così le analisi si possono rifare offline."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class FetchError(RuntimeError):
    pass


class HttpCache:
    def __init__(self, cache_dir: Path, max_age_hours: float = 12.0, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.max_age_s = max_age_hours * 3600
        self.offline = offline

    def _path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}{suffix}"

    def get_bytes(
        self, url: str, suffix: str = ".bin", headers: dict | None = None, max_age_hours: float | None = None
    ) -> bytes:
        path = self._path(url, suffix)
        max_age_s = self.max_age_s if max_age_hours is None else max_age_hours * 3600
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_s
        if path.exists() and (fresh or self.offline):
            return path.read_bytes()
        if self.offline:
            raise FetchError(f"Modalità offline e nessuna copia in cache per {url}")
        hdrs = {"User-Agent": USER_AGENT}
        hdrs.update(headers or {})
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=hdrs, timeout=30)
                resp.raise_for_status()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                return resp.content
            except requests.RequestException as exc:  # rete instabile: riprova
                last_exc = exc
                time.sleep(2**attempt)
        if path.exists():  # meglio un dato vecchio che nessun dato
            return path.read_bytes()
        raise FetchError(f"Download fallito per {url}: {last_exc}")
