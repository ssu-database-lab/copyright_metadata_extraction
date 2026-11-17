#!/usr/bin/env python3
"""Make an API key for one client of the public server (see api/web/api_keys.py).

    python3 api/tools/make_api_key.py muhayu

Prints the key ONCE (give it to the client through a private channel; it is not stored anywhere)
and the API_KEYS entry for the server (only the SHA-256 is stored there, so that line is not secret).
"""
import hashlib
import re
import secrets
import sys


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"[a-z0-9_-]{2,32}", sys.argv[1]):
        print("usage: make_api_key.py <client-name>   (lowercase letters, digits, - or _)", file=sys.stderr)
        return 2
    name = sys.argv[1]
    key = f"ck_{name}_{secrets.token_urlsafe(24)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    print(f"Key for {name} (shown once; give it to the client privately):\n  {key}\n")
    print("Server entry (not secret; several clients are joined with commas):")
    print(f"  API_KEYS={name}:{digest}")
    print("\nClient header:  Authorization: Bearer <key>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
