"""
Centralized environment variable loader.

All modules should use this instead of loading .env independently.
Searches for .env in order: project root → api/ → api/web/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_loaded = False

def load_env():
    """Load .env from the project root. Call is idempotent."""
    global _loaded
    if _loaded:
        return

    search_paths = [
        Path(__file__).parent.parent.parent / ".env",  # project root
        Path(__file__).parent.parent / ".env",          # api/
        Path(__file__).parent.parent / "web" / ".env",  # api/web/
    ]

    for env_path in search_paths:
        if env_path.exists():
            load_dotenv(env_path)
            _loaded = True
            return

    # No file found — rely on system env vars
    _loaded = True


# Auto-load on import
load_env()
