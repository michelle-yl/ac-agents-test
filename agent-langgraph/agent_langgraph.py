"""


Agentic RAG with LangGraph — Anthropic Claude for chat; local embeddings for retrieval.

    ANTHROPIC_API_KEY — required
    ANTHROPIC_CHAT_MODEL — default claude-3-5-sonnet-20241022 main agent + answer
    ANTHROPIC_GRADER_MODEL — default claude-3-5-haiku-20241022 relevance grader
    HF_EMBEDDING_MODEL — default sentence-transformers/all-MiniLM-L6-v2 (no extra API key)

Indexing (startup):
    SEED_URLS — optional comma-separated list of URLs; when unset, default seed URLs apply.
        Only these pages seed the searchable web corpus (no runtime URL ingestion).
    LOCAL_DOCS_DIR — optional directory scanned recursively at startup with Unstructured/local
        text fallback (.pdf/.docx/… plus .txt/.md/.csv/.json).

Unstructured (local files):
    UNSTRUCTURED_API_KEY — optional; when set, partition_via_api=true for UnstructuredLoader.
        When unset, local open-source partition runs via the unstructured library.

Caveman mode (https://github.com/juliusbrussee/caveman): terse replies, fewer output tokens.
    CAVEMAN_ENABLED=1 (default) | 0
    CAVEMAN_LEVEL=lite|full (default)|ultra|wenyan-lite|wenyan-full|wenyan-ultra

Install dependencies (recommended: use project venv + pinned lockfile):
    python -m venv .venv
    .\\.venv\\Scripts\\activate
    pip install -U pip
    pip install -r requirements.txt

In Cursor/VS Code, choose the workspace Python interpreter (defaults via .vscode/settings.json).

One-shot install (upgrade all LangChain peers together; avoids version skew):
    pip install -U langgraph langchain-anthropic langchain-huggingface sentence-transformers \
        langchain langchain-community langchain-text-splitters bs4 \
        langchain-unstructured unstructured-client "unstructured[pdf]" python-magic
"""


import getpass
import os
import re
import sys
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field

import langchain_core as _lc_core

# Fail fast: LangGraph stack needs langchain-core >= 1.4 (e.g. ModelProfile). Anaconda base
# often ships langchain_core 0.2.x — use project .venv (see .vscode/settings.json).
_lc_ver = getattr(_lc_core, "__version__", "0")
_lc_nums = [int(x) for x in re.findall(r"\d+", _lc_ver)[:3]] or [0]
while len(_lc_nums) < 3:
    _lc_nums.append(0)
if tuple(_lc_nums) < (1, 4, 0):
    _venv_py = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    raise RuntimeError(
        f"Incompatible langchain-core {_lc_ver} (need >= 1.4.0; includes ModelProfile). "
        f"Current interpreter: {sys.executable}\n"
        f"Run with project venv: {_venv_py} {Path(__file__).name}\n"
        "Or: python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
    )

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.messages import SystemMessage, convert_to_messages
from langchain_core.documents import Document

try:
    from langchain_unstructured import UnstructuredLoader
except ImportError:
    UnstructuredLoader = None  # type: ignore[misc, assignment]

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ── API key setup ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")


_set_env("ANTHROPIC_API_KEY")
# _set_env("UNSTRUCTURED_API_KEY")   # ← Unstructured API key (get one free at unstructured.io)

ANTHROPIC_CHAT_MODEL = os.environ.get(
    "ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6"
)
ANTHROPIC_GRADER_MODEL = os.environ.get(
    "ANTHROPIC_GRADER_MODEL", "claude-haiku-4-5"
)
HF_EMBEDDING_MODEL = os.environ.get(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

PROJECT_ROOT = Path(__file__).resolve().parent

_DEFAULT_SEED_URLS = (
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://www.subtraction.com/",
)

_LOCAL_DOC_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
        ".csv",
        ".html",
        ".htm",
        ".rst",
        ".json",
        ".xml",
        ".rtf",
        ".epub",
    }
)


def _seed_urls() -> list[str]:
    raw = os.environ.get("SEED_URLS", "").strip()
    if not raw:
        return list(_DEFAULT_SEED_URLS)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _local_docs_anchor() -> Path | None:
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


_LOCAL_DOCS_CONFIGURED_RAW = os.environ.get("LOCAL_DOCS_DIR", "").strip()

LOCAL_DOCS_ROOT = _local_docs_anchor()
if _LOCAL_DOCS_CONFIGURED_RAW and LOCAL_DOCS_ROOT is None:
    print(
        "[ingest] LOCAL_DOCS_DIR is set but not a usable directory; "
        "local startup ingest skipped.",
        file=sys.stderr,
    )


def _partition_via_api() -> bool:
    key = os.getenv("UNSTRUCTURED_API_KEY")
    return bool(key and key.strip())


def _is_under_anchor(path: Path, anchor: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(anchor.resolve(strict=False))
        return True
    except ValueError:
        return False


def _read_access_anchor_path() -> Path:
    """Local files opened via tools must resolve under LOCAL_DOCS_DIR if set."""
    return LOCAL_DOCS_ROOT if LOCAL_DOCS_ROOT is not None else PROJECT_ROOT


def resolve_user_local_path(raw: str) -> tuple[Path | None, str | None]:
    """Resolve ``raw`` against the read-access root; disallow path traversal."""
    anchor = _read_access_anchor_path()
    anchor_resolved = anchor.resolve(strict=False)
    candidate = Path(raw.strip()).expanduser()
    resolved = candidate if candidate.is_absolute() else (anchor / candidate).resolve(strict=False)

    try:
        if not _is_under_anchor(resolved, anchor_resolved):
            return (
                None,
                f"Path must be under {anchor_resolved} "
                "(set LOCAL_DOCS_DIR if you ingest from a docs folder)",
            )
    except (OSError, RuntimeError):
        return None, f"Could not normalize path: {raw!r}"

    if not resolved.is_file():
        return None, f"File not found: {resolved}"
    return resolved, None


def _mark_web_documents(docs: list[Document]) -> None:
    for d in docs:
        md = dict(d.metadata) if d.metadata else {}
        md.setdefault("source_type", "web")
        d.metadata = md


def load_web_seed_documents(url_list: list[str]) -> list[Document]:
    """Load starter web pages tagged as ``source_type=web``."""
    out: list[Document] = []
    for url in url_list:
        try:
            batch = WebBaseLoader(url).load()
        except Exception as exc:
            print(f"[ingest] WebBaseLoader skip {url!r}: {exc}", file=sys.stderr)
            continue
        _mark_web_documents(batch)
        out.extend(batch)
    return out


def _plain_text_documents(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Document(
            page_content=text,
            metadata={"source": str(path.resolve()), "source_type": "local"},
        )
    ]


def load_local_documents_from_path(path: Path) -> list[Document]:
    """Parse a single file with UnstructuredLoader or UTF-8 text fallback."""
    if UnstructuredLoader is None:
        if path.suffix.lower() in {".txt", ".md", ".csv", ".json"}:
            return _plain_text_documents(path)
        print(f"[ingest] skip non-text without Unstructured: {path}", file=sys.stderr)
        return []

    loader = UnstructuredLoader(
        file_path=str(path),
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=_partition_via_api(),
    )
    try:
        loaded = loader.load()
    except Exception as exc:
        print(f"[ingest] UnstructuredLoader failed on {path}: {exc}", file=sys.stderr)
        loaded = []

    if not loaded and path.suffix.lower() in {".txt", ".md", ".csv", ".json"}:
        return _plain_text_documents(path)

    for d in loaded:
        md = dict(d.metadata) if d.metadata else {}
        md.setdefault("source", str(path.resolve()))
        md.setdefault("source_type", "local")
        d.metadata = md

    return loaded


def ingest_local_documents_from_directory(anchor: Path) -> list[Document]:
    """Recursively load supported files beneath ``anchor`` (path-safe)."""
    anchor_resolved = anchor.resolve(strict=False)
    seen: set[str] = set()
    out: list[Document] = []

    for p in anchor_resolved.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _LOCAL_DOC_EXTENSIONS:
            continue

        rp = p.resolve(strict=False)
        if not _is_under_anchor(rp, anchor_resolved):
            continue

        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        out.extend(load_local_documents_from_path(p))

    return out


def build_startup_corpus() -> list[Document]:
    """Combine seeded web URLs and optional ``LOCAL_DOCS_DIR`` ingest."""
    web_docs = load_web_seed_documents(_seed_urls())
    local_docs: list[Document] = []
    if LOCAL_DOCS_ROOT is not None:
        local_docs = ingest_local_documents_from_directory(LOCAL_DOCS_ROOT)

    merged = [*web_docs, *local_docs]
    if web_docs:
        url_line = ", ".join(_seed_urls())
        print(
            f"[ingest] web docs: {len(web_docs)} loaded parts from URLs ({url_line})",
            file=sys.stderr,
        )
    if LOCAL_DOCS_ROOT is not None:
        print(
            f"[ingest] local docs: {len(local_docs)} loaded parts under {LOCAL_DOCS_ROOT}",
            file=sys.stderr,
        )

    return merged


# ── Caveman output style (github.com/juliusbrussee/caveman, skills/caveman/SKILL.md) ─
def _env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


CAVEMAN_ENABLED = _env_truthy("CAVEMAN_ENABLED", "1")
CAVEMAN_LEVEL = os.environ.get("CAVEMAN_LEVEL", "full").strip().lower()

# Intensity rules — keep in sync with upstream SKILL.md
_CAVEMAN_INTENSITY: dict[str, str] = {
    "lite": (
        "Output style: no filler or hedging. Keep articles and full sentences. "
        "Professional but tight. Technical terms exact; leave code blocks unchanged."
    ),
    "full": (
        "Output style: respond terse like smart caveman. All technical substance stay; fluff die. "
        "Drop articles (a/an/the), filler (just/really/basically/actually/simply), "
        "pleasantries, hedging. Fragments OK. Short synonyms. Technical terms exact; "
        "code blocks unchanged. Pattern: [thing] [action] [reason]. [next step]."
    ),
    "ultra": (
        "Output style: ultra-terse. Abbreviate prose words (DB/auth/config/req/res/fn/impl), "
        "strip conjunctions, use arrows for causality (X → Y). "
        "Never abbreviate code symbols, function names, API names, error strings."
    ),
    "wenyan-lite": (
        "Output style: semi-classical Chinese register. Drop filler/hedging; "
        "keep clearer grammar. Technical tokens (API/code) stay as in source."
    ),
    "wenyan-full": (
        "Output style: maximum classical Chinese terseness (文言文). "
        "Classical particles, subjects often omitted where clear."
    ),
    "wenyan-ultra": (
        "Output style: extreme classical abbreviation; maximum compression; still precise."
    ),
}


def _caveman_system_text() -> str:
    if not CAVEMAN_ENABLED:
        return ""
    intensity = _CAVEMAN_INTENSITY.get(CAVEMAN_LEVEL, _CAVEMAN_INTENSITY["ultra"])
    return (
        f"You are a helpful assistant.\n\n{intensity}\n\n"
        "ACTIVE on every assistant reply in this chat unless user says "
        '"stop caveman" or "normal mode".\n'
        "When security, irreversible actions, or ambiguity would suffer from terseness, "
        "switch to clear full sentences for that part only, then resume terse style.\n"
    )


def _with_caveman(messages):
    """Prepend caveman system message for user-visible model calls."""
    chain = convert_to_messages(messages)
    extra = _caveman_system_text()
    if not extra:
        return chain
    return [SystemMessage(content=extra), *chain]


# ── Startup corpus → embeddings ─────────────────────────────────────────────
_startup_documents = build_startup_corpus()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=20,
)
document_chunks = text_splitter.split_documents(_startup_documents)

# Anthropic has no embeddings API — use Hugging Face sentence-transformers locally.
_embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vectorstore = InMemoryVectorStore.from_documents(
    documents=document_chunks,
    embedding=_embeddings,
)
retriever = vectorstore.as_retriever()


@tool
def search_knowledge_base(query: str) -> str:
    """Semantic search over indexed content from startup SEED_URLS pages plus LOCAL_DOCS_DIR.

    Use this tool for gist questions and keyword/semantic lookups. Prefer
    load_document only when you need full text from a known file path.
    """
    ranked = retriever.invoke(query)
    return "\n\n".join(part.page_content for part in ranked)


search_knowledge_tool = search_knowledge_base


@tool
def load_document(file_path: str) -> str:
    """Load full extracted text from a single local document.

    Path resolves under LOCAL_DOCS_DIR when that env var is set; otherwise under
    the project root only. Prefer search_knowledge_base for scoped questions.
    Uses Unstructured (API if UNSTRUCTURED_API_KEY is set, else local parsing).
    """
    resolved_path, guard_error = resolve_user_local_path(file_path)
    if guard_error:
        return guard_error

    pieces = load_local_documents_from_path(resolved_path)
    if not pieces:
        return (
            "No extractable content from "
            f"{resolved_path}; check format or unstructured install/deps."
        )
    return "\n\n".join(doc.page_content for doc in pieces)


document_loader_tool = load_document


# ── Agent query / tools wiring ───────────────────────────────────────────────
response_model = init_chat_model(
    ANTHROPIC_CHAT_MODEL, model_provider="anthropic", temperature=0
)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide whether to call ``search_knowledge_base`` or
    ``load_document``, or simply respond to the user without tools.
    """
    response = response_model.bind_tools(
        [search_knowledge_tool, document_loader_tool]
    ).invoke(
        _with_caveman(state["messages"])
    )
    return {"messages": [response]}


# ── 4. Grade documents ────────────────────────────────────────────────────────
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant "
    "to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model(
    ANTHROPIC_GRADER_MODEL, model_provider="anthropic", temperature=0
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


# ── 5. Rewrite question ───────────────────────────────────────────────────────
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke(
        _with_caveman([{"role": "user", "content": prompt}])
    )
    return {"messages": [HumanMessage(content=response.content)]}


# ── 6. Generate an answer ─────────────────────────────────────────────────────
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke(
        _with_caveman([{"role": "user", "content": prompt}])
    )
    return {"messages": [response]}


# ── 7. Assemble the graph ─────────────────────────────────────────────────────
workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node(
    "retrieve", ToolNode([search_knowledge_tool, document_loader_tool])
)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


# ── 8. Run the agentic RAG ────────────────────────────────────────────────────
if __name__ == "__main__":
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "In her PHY180 pendulum report, which method of finding Q for a pendulum did Michelle Liu say was better?",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")