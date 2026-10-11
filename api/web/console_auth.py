"""시험 콘솔 접근 제어 — 비밀번호 해시 · 서명 세션 쿠키 · 로그인 시도 제한 · 계정 설정 CLI.

설정은 저장소 루트의 `.env` 세 줄이다:
  CONSOLE_USER           로그인 아이디 (기본 admin)
  CONSOLE_PASSWORD_HASH  pbkdf2_sha256$<반복수>$<salt_b64>$<hash_b64>   (반복수 ≥ 200,000)
  CONSOLE_SECRET         세션 쿠키 서명 키 (hex, 32바이트 이상)

계정 설정 (비밀번호·해시는 화면에 찍지 않는다):
  python -m api.web.console_auth set-password --user admin            # 비밀번호를 표준입력에서 읽는다
  python -m api.web.console_auth set-password --user admin --password '…' [--env-file PATH]

**실행 중인 서버도 .env 를 다시 읽는다** — 파일이 바뀌면(mtime) 다음 요청부터 새 값을 쓴다. 단, 서버를 띄울 때
.env 와 **다른 값**을 환경변수로 직접 준 경우(systemd Environment=, 시험용 덮어쓰기)에는 그 환경변수가 우선한다.

세션 쿠키 `console_session` = base64url("user|만료시각") + "." + HMAC-SHA256 서명. 서명 키는 CONSOLE_SECRET 과
비밀번호 해시로 만든다 — 비밀번호를 바꾸면 이전 세션이 모두 끊긴다. 수명 12시간, HttpOnly, SameSite=Lax,
HTTPS 요청일 때만 Secure.

이 모듈은 가볍다(표준 라이브러리 + python-dotenv). `import api` 도 무거운 스택을 올리지 않으므로 CLI 가 즉시 뜬다.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import logging
import os
import re
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# api/web/console_auth.py → parents[2] = 저장소 루트. resolve() 는 /home/… 심볼릭 링크를 /mnt/c 로 풀어 버리므로 abspath.
REPO_ROOT = Path(os.path.abspath(__file__)).parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

KEY_USER = "CONSOLE_USER"
KEY_HASH = "CONSOLE_PASSWORD_HASH"
KEY_SECRET = "CONSOLE_SECRET"
_KEYS = (KEY_USER, KEY_HASH, KEY_SECRET)
DEFAULT_USER = "admin"

ALGO = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 310_000
MIN_ITERATIONS = 200_000
MIN_PASSWORD_LEN = 8

SESSION_COOKIE = "console_session"
SESSION_LIFETIME_SEC = 12 * 3600

# 로그인 시도 제한: 10분 안에 5번 틀리면, 그 뒤 시도는 마지막 실패로부터 30초를 기다린다.
THROTTLE_MAX_FAILURES = 5
THROTTLE_WINDOW_SEC = 600
THROTTLE_LOCKOUT_SEC = 30
FAIL_DELAY_SEC = 0.5          # 실패 응답은 고정 지연 뒤에 돌려준다 (테스트에서 줄일 수 있다)

SET_PASSWORD_HINT = "python -m api.web.console_auth set-password --user admin"


def _now() -> float:
    """현재 시각 — 테스트가 바꿔 끼운다."""
    return time.time()


# ---------------------------------------------------------------------------
# 비밀번호 해시
# ---------------------------------------------------------------------------
def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS, salt: Optional[bytes] = None) -> str:
    """pbkdf2_sha256$<반복수>$<salt_b64>$<hash_b64>."""
    if iterations < MIN_ITERATIONS:
        raise ValueError(f"반복수는 {MIN_ITERATIONS:,} 이상이어야 합니다")
    salt = salt if salt is not None else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGO}${iterations}${_b64(salt)}${_b64(dk)}"


def parse_hash(encoded: Optional[str]) -> Optional[tuple]:
    """(반복수, salt, hash) 또는 None(형식 오류)."""
    if not encoded:
        return None
    parts = str(encoded).strip().split("$")
    if len(parts) != 4 or parts[0] != ALGO:
        return None
    try:
        it = int(parts[1])
        salt, dk = _unb64(parts[2]), _unb64(parts[3])
    except (ValueError, TypeError):
        return None
    if it < 1 or not salt or not dk:
        return None
    return it, salt, dk


def verify_password(password: str, encoded: Optional[str]) -> bool:
    p = parse_hash(encoded)
    if p is None:
        return False
    it, salt, dk = p
    got = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, it)
    return hmac.compare_digest(got, dk)


# ---------------------------------------------------------------------------
# 설정 읽기 (.env 실시간 + 환경변수 덮어쓰기)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    user: str
    password_hash: Optional[str]
    secret: Optional[str]
    ephemeral_secret: bool = False

    @property
    def configured(self) -> bool:
        return parse_hash(self.password_hash) is not None


_ENV_PATH: Path = DEFAULT_ENV_PATH
_STARTUP_FILE: Dict[str, Optional[str]] = {}
_FILE_CACHE: Dict[str, object] = {"sig": None, "values": {}}
_CFG_LOCK = threading.Lock()
_EPHEMERAL_SECRET = secrets.token_hex(32)
_WARNED_EPHEMERAL = False


def _file_values() -> Dict[str, Optional[str]]:
    """.env 의 CONSOLE_* 값 — 파일 mtime·크기가 같으면 캐시를 쓴다."""
    p = _ENV_PATH
    try:
        st = p.stat()
    except OSError:
        return {}
    sig = (str(p), st.st_mtime_ns, st.st_size)
    with _CFG_LOCK:
        if _FILE_CACHE["sig"] == sig:
            return dict(_FILE_CACHE["values"])  # type: ignore[arg-type]
    try:
        from dotenv import dotenv_values
        raw = dotenv_values(p)
    except Exception as e:                     # 읽기 실패가 로그인 화면을 죽이면 안 된다
        logger.warning("콘솔 설정 파일을 읽지 못했습니다 (%s): %s", p, type(e).__name__)
        return {}
    vals = {k: (raw.get(k) or None) for k in _KEYS if k in raw}
    with _CFG_LOCK:
        _FILE_CACHE["sig"], _FILE_CACHE["values"] = sig, vals
    return dict(vals)


def configure(env_path: Optional[os.PathLike] = None) -> None:
    """설정 파일 위치를 정하고, 지금의 환경변수가 그 파일에서 온 것인지 구분할 기준을 남긴다.

    서버 기동 시 app.py 가 load_dotenv 로 .env 를 환경변수에 올려 둔다. 그 값은 파일과 같으므로 '파일에서 온 값' 으로
    보고 이후 파일이 바뀌면 파일을 따른다. 파일과 다른 환경변수는 명시적 덮어쓰기로 본다.
    """
    global _ENV_PATH, _STARTUP_FILE
    if env_path is not None:
        _ENV_PATH = Path(env_path)
    with _CFG_LOCK:
        _FILE_CACHE["sig"], _FILE_CACHE["values"] = None, {}
    _STARTUP_FILE = _file_values()


def env_path() -> Path:
    return _ENV_PATH


def _setting(key: str) -> Optional[str]:
    env = (os.environ.get(key) or "").strip() or None
    fv = _file_values().get(key)
    if env is not None and env == _STARTUP_FILE.get(key):
        return fv                              # 기동 때 .env 에서 올라온 값 → 지금의 파일을 따른다
    if env is not None:
        return env                             # 파일과 다른 명시적 환경변수
    return fv


def get_settings() -> Settings:
    global _WARNED_EPHEMERAL
    user = _setting(KEY_USER) or DEFAULT_USER
    pw_hash = _setting(KEY_HASH)
    secret = _setting(KEY_SECRET)
    ephemeral = False
    if not secret or len(secret) < 32:
        # 해시만 있고 서명 키가 없으면(수동 편집) 프로세스 임시 키로 서명한다 — 재시작하면 다시 로그인해야 한다.
        if pw_hash and not _WARNED_EPHEMERAL:
            logger.warning("CONSOLE_SECRET 이 없거나 짧아 임시 서명 키를 씁니다 — 서버를 재시작하면 세션이 끊깁니다 "
                           "(%s 로 다시 설정하세요)", SET_PASSWORD_HINT)
            _WARNED_EPHEMERAL = True
        secret, ephemeral = _EPHEMERAL_SECRET, True
    return Settings(user=user, password_hash=pw_hash, secret=secret, ephemeral_secret=ephemeral)


def check_credentials(settings: Settings, user: str, password: str) -> bool:
    """아이디·비밀번호 확인 — 아이디가 달라도 해시 계산은 한다(응답 시간으로 아이디를 가늠하지 못하게)."""
    pw_ok = verify_password(password or "", settings.password_hash)
    user_ok = hmac.compare_digest((user or "").strip().encode("utf-8"), settings.user.encode("utf-8"))
    return bool(pw_ok and user_ok)


# ---------------------------------------------------------------------------
# 세션 쿠키
# ---------------------------------------------------------------------------
def _signing_key(settings: Settings) -> bytes:
    material = f"{settings.secret}|{settings.password_hash or ''}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64url(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def make_session_token(settings: Settings, user: str, now: Optional[float] = None,
                       lifetime: int = SESSION_LIFETIME_SEC) -> str:
    exp = int((now if now is not None else _now()) + lifetime)
    payload = f"{user}|{exp}".encode("utf-8")
    sig = hmac.new(_signing_key(settings), payload, hashlib.sha256).digest()
    return f"{_b64url(payload)}.{_b64url(sig)}"


def read_session_token(settings: Settings, token: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """서명·만료·아이디를 확인하고 아이디를 돌려준다. 하나라도 어긋나면 None."""
    if not token or "." not in token or not settings.configured:
        return None
    p64, s64 = token.split(".", 1)
    try:
        payload = _unb64url(p64)
        sig = _unb64url(s64)
    except (ValueError, TypeError):
        return None
    want = hmac.new(_signing_key(settings), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, want):
        return None
    try:
        user, exp_s = payload.decode("utf-8").rsplit("|", 1)
        exp = int(exp_s)
    except (ValueError, UnicodeDecodeError):
        return None
    if exp <= (now if now is not None else _now()):
        return None
    if not hmac.compare_digest(user.encode("utf-8"), settings.user.encode("utf-8")):
        return None
    return user


def is_https(request) -> bool:
    if getattr(request.url, "scheme", "") == "https":
        return True
    return (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower() == "https"


def session_user(request, settings: Optional[Settings] = None) -> Optional[str]:
    s = settings or get_settings()
    return read_session_token(s, request.cookies.get(SESSION_COOKIE))


def set_session_cookie(response, request, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_LIFETIME_SEC, path="/",
                        httponly=True, samesite="lax", secure=is_https(request))


def clear_session_cookie(response, request=None) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax",
                           secure=bool(request is not None and is_https(request)))


def safe_next(target: Optional[str], default: str = "/console") -> str:
    """로그인 뒤 이동할 곳 — 같은 사이트의 경로만 받는다 (열린 리다이렉트 방지)."""
    t = (target or "").strip()
    if not t or not t.startswith("/") or t.startswith("//") or "\\" in t:
        return default
    if any(ord(c) < 32 or ord(c) == 127 for c in t):
        return default
    parts = urlsplit(t)
    if parts.scheme or parts.netloc:
        return default
    return t


# ---------------------------------------------------------------------------
# 로그인 시도 제한
# ---------------------------------------------------------------------------
class LoginThrottle:
    """IP 별 실패 기록 — 창(10분) 안에 실패가 max 번 이상이면 마지막 실패로부터 lockout 초 동안 거절한다."""

    def __init__(self, max_failures: int = THROTTLE_MAX_FAILURES, window: float = THROTTLE_WINDOW_SEC,
                 lockout: float = THROTTLE_LOCKOUT_SEC):
        self.max_failures, self.window, self.lockout = max_failures, window, lockout
        self._fails: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, ip: str, now: float) -> Deque[float]:
        dq = self._fails.setdefault(ip, deque())
        while dq and now - dq[0] > self.window:
            dq.popleft()
        return dq

    def retry_after(self, ip: str, now: Optional[float] = None) -> float:
        """기다려야 할 초 (0 이면 시도 가능)."""
        now = _now() if now is None else now
        with self._lock:
            dq = self._prune(ip, now)
            if len(dq) >= self.max_failures:
                wait = self.lockout - (now - dq[-1])
                return max(0.0, wait)
            return 0.0

    def record_failure(self, ip: str, now: Optional[float] = None) -> None:
        now = _now() if now is None else now
        with self._lock:
            self._prune(ip, now).append(now)

    def reset(self, ip: Optional[str] = None) -> None:
        with self._lock:
            if ip is None:
                self._fails.clear()
            else:
                self._fails.pop(ip, None)


THROTTLE = LoginThrottle()


# ---------------------------------------------------------------------------
# .env 갱신 + CLI
# ---------------------------------------------------------------------------
def _env_line_key(line: str) -> Optional[str]:
    m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    return m.group(1) if m else None


def update_env_file(path: os.PathLike, updates: Dict[str, str]) -> None:
    """주어진 키만 바꾸거나 덧붙이고 나머지 줄은 그대로 둔다 (같은 키가 여러 줄이면 첫 줄만 남긴다)."""
    p = Path(path)
    raw = p.read_bytes().decode("utf-8") if p.exists() else ""     # read_text 는 줄바꿈을 \n 으로 바꿔 버린다
    nl = "\r\n" if "\r\n" in raw else "\n"          # Windows 에서 만든 .env 는 줄바꿈을 그대로 둔다
    lines = raw.splitlines()
    out, seen = [], set()
    for line in lines:
        k = _env_line_key(line)
        if k in updates:
            if k in seen:
                continue
            out.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out.append(line)
    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# 시험 콘솔 계정 (python -m api.web.console_auth set-password 로 갱신)")
        out.extend(f"{k}={updates[k]}" for k in missing)
    tmp = p.with_name(p.name + ".tmp_console")
    tmp.write_bytes((nl.join(out) + nl).encode("utf-8"))
    try:
        if p.exists():
            os.chmod(tmp, p.stat().st_mode & 0o777)
        else:
            os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, p)


def _read_password(args) -> str:
    if args.password is not None:
        return args.password
    if sys.stdin is not None and sys.stdin.isatty():
        pw1 = getpass.getpass("새 비밀번호: ")
        pw2 = getpass.getpass("비밀번호 확인: ")
        if pw1 != pw2:
            raise SystemExit("⛔ 두 비밀번호가 다릅니다")
        return pw1
    line = sys.stdin.readline() if sys.stdin is not None else ""
    return line.rstrip("\r\n")


def set_password(env_file: os.PathLike, user: str, password: str) -> Dict[str, bool]:
    """세 줄을 쓰고 {"secret_created": bool} 를 돌려준다 (비밀번호·해시는 돌려주지 않는다)."""
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"비밀번호는 {MIN_PASSWORD_LEN}자 이상이어야 합니다")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,64}", user or ""):
        raise ValueError("아이디는 영문·숫자·_ . @ - 로 64자 이내여야 합니다")
    p = Path(env_file)
    existing = {}
    if p.exists():
        try:
            from dotenv import dotenv_values
            existing = dotenv_values(p)
        except Exception:
            existing = {}
    secret = (existing.get(KEY_SECRET) or "").strip()
    created = len(secret) < 32 or not re.fullmatch(r"[0-9a-fA-F]+", secret)
    if created:
        secret = secrets.token_hex(32)
    # 작은따옴표 — python-dotenv 가 '$' 를 변수로 풀지 않게 한다
    update_env_file(p, {KEY_USER: user, KEY_HASH: f"'{hash_password(password)}'", KEY_SECRET: secret})
    return {"secret_created": created}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m api.web.console_auth",
                                 description="시험 콘솔 계정 설정 — .env 의 CONSOLE_USER · CONSOLE_PASSWORD_HASH · CONSOLE_SECRET")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("set-password", help="콘솔 아이디·비밀번호를 설정한다")
    sp.add_argument("--user", default=DEFAULT_USER, help="로그인 아이디 (기본 admin)")
    sp.add_argument("--password", default=None, help="비밀번호 (생략하면 표준입력에서 읽는다)")
    sp.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="설정 파일 (기본: 저장소 루트 .env)")
    args = ap.parse_args(argv)
    if args.cmd != "set-password":
        ap.print_help()
        return 2
    try:
        password = _read_password(args)
        info = set_password(args.env_file, args.user, password)
    except ValueError as e:
        print(f"⛔ {e}", file=sys.stderr)
        return 2
    print(f"콘솔 계정을 저장했습니다 — 아이디 {args.user} · 파일 {args.env_file}")
    print(f"  {KEY_USER} · {KEY_HASH} · {KEY_SECRET}"
          f"({'새로 만듦' if info['secret_created'] else '기존 값 유지'})")
    print("  실행 중인 서버는 다음 요청부터 새 비밀번호를 씁니다 (기존 세션은 끊깁니다).")
    return 0


configure()

if __name__ == "__main__":
    sys.exit(main())
