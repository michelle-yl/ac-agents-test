"""Load configuration from sdl-agents/.env (optional repo-root .env first)."""

from __future__ import annotations

import os
from pathlib import Path

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
OPENCLAW_BASE_URL = os.environ.get("OPENCLAW_BASE_URL", "http://localhost:18789").rstrip(
    "/"
)
SAFETY_DOCS_DIR = Path(
    os.environ.get("SAFETY_DOCS_DIR", str(SDL_AGENTS_ROOT / "corpus" / "safety"))
)
PROCEDURES_DOCS_DIR = Path(
    os.environ.get(
        "PROCEDURES_DOCS_DIR", str(SDL_AGENTS_ROOT / "corpus" / "procedures")
    )
)
RESEARCH_DOCS_DIR = Path(
    os.environ.get("RESEARCH_DOCS_DIR", str(SDL_AGENTS_ROOT / "corpus" / "research"))
)
INDICES_DIR = SDL_AGENTS_ROOT / "indices"
MAX_RESULT_ROWS = int(os.environ.get("DB_AGENT_MAX_ROWS", "500"))
DB_TIMEOUT_SECONDS = float(os.environ.get("DB_AGENT_TIMEOUT", "5"))
