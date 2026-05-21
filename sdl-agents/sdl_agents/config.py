"""Load configuration from sdl-agents/.env (optional repo-root .env first)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SDL_AGENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ENV = REPO_ROOT / ".env"
LOCAL_ENV = SDL_AGENTS_ROOT / ".env"

_DEFAULT_SEED_URLS = (
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://www.subtraction.com/",
)

SUPPORTED_DOC_EXTENSIONS = frozenset(
    {".md", ".txt", ".csv", ".json", ".pdf", ".docx"}
)


def load_config() -> None:
    if REPO_ENV.is_file():
        load_dotenv(REPO_ENV)
    if LOCAL_ENV.is_file():
        load_dotenv(LOCAL_ENV, override=True)


load_config()

# Pytest forces mock integration when SDL_TEST_USE_MOCK_INTEGRATION=1 (see tests/conftest.py)
if os.environ.get("SDL_TEST_USE_MOCK_INTEGRATION") == "1":
    os.environ["SDL_INTEGRATION_MODE"] = "mock"


def database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    user = os.environ.get("POSTGRES_USER", "angie")
    password = os.environ.get("POSTGRES_PASSWORD", "angie")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "angie_monitoring_replica")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def integration_mode() -> str:
    return os.environ.get("SDL_INTEGRATION_MODE", "mock").strip().lower()


def is_live_integration() -> bool:
    return integration_mode() == "live"


def normalize_hermes_openai_base(raw: str) -> str:
    """Hermes API server OpenAI root must be …/v1 (see Nous Hermes API Server docs)."""
    u = raw.strip().rstrip("/")
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


def hermes_http_origin(openai_base: str) -> str:
    """Scheme + host + port for GET /health (strip trailing /v1)."""
    return openai_base.removesuffix("/v1")


def seed_urls() -> list[str]:
    raw = os.environ.get("SEED_URLS", "").strip()
    if not raw:
        return list(_DEFAULT_SEED_URLS)
    return [u.strip() for u in raw.split(",") if u.strip()]


def local_docs_dir() -> Path | None:
    raw = os.environ.get("LOCAL_DOCS_DIR", "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    return resolved


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
ANTHROPIC_GRADER_MODEL = os.environ.get("ANTHROPIC_GRADER_MODEL", "claude-haiku-4-5")
HF_EMBEDDING_MODEL = os.environ.get(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
HERMES_BASE_URL = normalize_hermes_openai_base(
    os.environ.get("HERMES_BASE_URL", "http://127.0.0.1:8642")
)
HERMES_API_KEY = os.environ.get("HERMES_API_KEY", "")
HERMES_MODEL = os.environ.get("HERMES_MODEL", "hermes-agent")
HERMES_CONNECT_TIMEOUT = float(os.environ.get("HERMES_CONNECT_TIMEOUT", "10"))
HERMES_READ_TIMEOUT = float(os.environ.get("HERMES_READ_TIMEOUT", "180"))

SAFETY_DOCS_DIR = Path(
    os.environ.get("SAFETY_DOCS_DIR", str(SDL_AGENTS_ROOT / "corpus" / "safety"))
)
PROCEDURES_DOCS_DIR = Path(
    os.environ.get(
        "PROCEDURES_DOCS_DIR", str(SDL_AGENTS_ROOT / "corpus" / "procedures")
    )
)
_default_academic = str(SDL_AGENTS_ROOT / "corpus" / "academic")
ACADEMIC_DOCS_DIR = Path(
    os.environ.get("ACADEMIC_DOCS_DIR", os.environ.get("RESEARCH_DOCS_DIR", _default_academic))
)
# Backward-compatible alias (prefer ACADEMIC_DOCS_DIR in new config)
RESEARCH_DOCS_DIR = ACADEMIC_DOCS_DIR
INDICES_DIR = SDL_AGENTS_ROOT / "indices"
MAX_RESULT_ROWS = int(os.environ.get("DB_AGENT_MAX_ROWS", "500"))
DB_TIMEOUT_SECONDS = float(os.environ.get("DB_AGENT_TIMEOUT", "5"))

DB_DIR = REPO_ROOT / "db"
DEFAULT_MONITOR_JSON_DIR = DB_DIR / "seed" / "data" / "monitoring"


def _env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


HERMES_WSL_AUTO = _env_truthy("HERMES_WSL_AUTO", "1")
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu").strip()


def discover_wsl_ipv4(distro: str | None = None) -> str | None:
    """Best-effort WSL2 eth0 IPv4 (Windows host → WSL services)."""
    if sys.platform != "win32":
        return None
    name = (distro or WSL_DISTRO).strip()
    if not name:
        return None
    try:
        proc = subprocess.run(
            ["wsl", "-d", name, "--", "ip", "-4", "-o", "addr", "show", "eth0"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", proc.stdout)
    return match.group(1) if match else None


def _localhost_wsl_fallback_urls(
    primary: str,
    *,
    default_port: int,
    wsl_auto: bool,
) -> list[str]:
    """Append WSL eth0 host URL when primary uses localhost on Windows."""
    urls = [primary.rstrip("/")]
    if not wsl_auto or sys.platform != "win32":
        return urls
    parsed = urlparse(urls[0])
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        return urls
    wsl_ip = discover_wsl_ipv4()
    if not wsl_ip:
        return urls
    port = parsed.port or default_port
    fallback = f"http://{wsl_ip}:{port}"
    if fallback not in urls:
        urls.append(fallback)
    return urls


def hermes_openai_base_urls() -> list[str]:
    """Hermes OpenAI API roots (…/v1) with optional WSL fallback on Windows."""
    origin = hermes_http_origin(HERMES_BASE_URL)
    bases = _localhost_wsl_fallback_urls(origin, default_port=8642, wsl_auto=HERMES_WSL_AUTO)
    return [normalize_hermes_openai_base(b) for b in bases]


def monitor_json_dir() -> Path:
    raw = os.environ.get("MONITOR_JSON_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_MONITOR_JSON_DIR.resolve()


MONITOR_INGEST_INTERVAL_SEC = int(os.environ.get("MONITOR_INGEST_INTERVAL_SEC", "120"))
MONITOR_POLL_INTERVAL_SEC = int(os.environ.get("MONITOR_POLL_INTERVAL_SEC", "60"))
MONITOR_CACHE_MAX_AGE_SEC = int(os.environ.get("MONITOR_CACHE_MAX_AGE_SEC", "120"))

DB_FAST_PATH_ENABLED = _env_truthy("DB_FAST_PATH_ENABLED", "1")
ROUTER_KEYWORD_FIRST = _env_truthy("ROUTER_KEYWORD_FIRST", "1")
DB_AGENT_MODEL = os.environ.get("DB_AGENT_MODEL", ANTHROPIC_GRADER_MODEL)
DB_FAST_PATH_MAX_ROWS = int(os.environ.get("DB_FAST_PATH_MAX_ROWS", "100"))
