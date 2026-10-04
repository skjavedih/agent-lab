"""Singleton Playwright browser instance using persistent context.

Why persistent context?
A persistent context stores cookies, localStorage, and session data in
`user_data_dir` across runs. This means the agent can reuse a logged-in
browser session (e.g., Gmail) without re-authenticating on every startup.

Usage
-----
    from app.browser import get_browser, get_page

    page = get_page()          # ready-to-use Page object
    page.goto("https://...")
"""

from __future__ import annotations

import atexit
import threading
from pathlib import Path

from playwright.sync_api import (
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Profile directory is relative to the project root. Using a dot-prefixed name
# keeps it out of the way but visible when needed. Add to .gitignore to avoid
# accidentally committing session cookies to source control.
_USER_DATA_DIR = str(Path(__file__).resolve().parent.parent / ".browser_profile")

# ---------------------------------------------------------------------------
# Internal singleton state (module-level, protected by a lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_playwright: Playwright | None = None
_context: BrowserContext | None = None
_page: Page | None = None


def _start() -> None:
    """Lazily start Playwright and open a persistent browser context."""
    global _playwright, _context, _page

    _playwright = sync_playwright().start()

    _context = _playwright.chromium.launch_persistent_context(
        user_data_dir=_USER_DATA_DIR,
        headless=False,
        args=["--no-sandbox"],
    )

    # Reuse the first tab that the browser opens, or create a new one.
    _page = _context.pages[0] if _context.pages else _context.new_page()


def _shutdown() -> None:
    """Close the browser and stop Playwright. Registered with atexit."""
    global _playwright, _context, _page
    if _context is not None:
        try:
            _context.close()
        except Exception:
            pass
        _context = None
    if _playwright is not None:
        try:
            _playwright.stop()
        except Exception:
            pass
        _playwright = None
    _page = None


atexit.register(_shutdown)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_context() -> BrowserContext:
    """Return the singleton :class:`BrowserContext`, starting Playwright if needed."""
    global _context
    with _lock:
        if _context is None:
            _start()
    return _context  # type: ignore[return-value]


def get_page() -> Page:
    """Return a ready-to-use :class:`Page` from the persistent context.

    The same page object is reused across calls. If the page has been closed
    externally (e.g., user closed the tab), a fresh page is opened.
    """
    global _page
    with _lock:
        if _context is None:
            _start()
        # Re-open page if it was closed
        if _page is None or _page.is_closed():
            _page = _context.new_page()  # type: ignore[union-attr]
    return _page  # type: ignore[return-value]
