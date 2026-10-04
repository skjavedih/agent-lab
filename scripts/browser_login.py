"""One-time login helper script.

Run this script once to authenticate in the browser. Your session cookies and
storage will be saved to .browser_profile so the agent can reuse them in
future runs without prompting you again.

Usage
-----
    python scripts/browser_login.py

The browser window will open https://mail.google.com. Sign in normally, then
come back to this terminal and press Enter to save the session and close.
"""

import sys
from pathlib import Path

# Make sure the project root is on the path so `app.browser` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.browser import get_context, get_page  # noqa: E402  (after sys.path fix)


def main() -> None:
    print("Opening https://mail.google.com …")
    print("Please log in inside the browser window that appears.")
    print()

    page = get_page()
    page.goto("https://mail.google.com", wait_until="domcontentloaded")

    input("Press Enter here once you have finished logging in to save your session…\n")

    # Closing the context flushes the profile to disk (cookies, localStorage, etc.)
    ctx = get_context()
    ctx.close()

    print("Session saved to .browser_profile/")
    print("You can now run the main agent; it will stay logged in automatically.")


if __name__ == "__main__":
    main()
