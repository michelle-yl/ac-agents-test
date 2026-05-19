"""LangGraph agentic RAG for general knowledge (web + local docs)."""

from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from sdl_agents.agents.knowledge.ingest import (
    build_startup_corpus,
    resolve_user_local_path,
)
from sdl_agents.caveman import with_caveman
from sdl_agents.config import ANTHROPIC_CHAT_MODEL, ANTHROPIC_GRADER_MODEL, HF_EMBEDDING_MODEL
from sdl_agents.integrations.documents import load_file, to_langchain_documents

_retriever = None
_response_model = None
_grader_model = None


def _ensure_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    docs = build_startup_corpus()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(docs) if docs else []
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    store = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)
    _retriever = store.as_retriever()
    return _retriever


def _model():
    global _response_model
    if _response_model is None:
        _response_model = init_chat_model(
            ANTHROPIC_CHAT_MODEL, model_provider="anthropic", temperature=0
        )
    return _response_model


def _grader():
    global _grader_model
    if _grader_model is None:
        _grader_model = init_chat_model(
            ANTHROPIC_GRADER_MODEL, model_provider="anthropic", temperature=0
        )
    return _grader_model


@tool
def search_knowledge_base(query: str) -> str:
    """Semantic search over SEED_URLS and LOCAL_DOCS_DIR indexed at startup."""
    ranked = _ensure_retriever().invoke(query)
    return "\n\n".join(part.page_content for part in ranked)


@tool
def load_document(file_path: str) -> str:
    """Load full text from a local file (.md .txt .csv .json .pdf .docx).

    Path must be under LOCAL_DOCS_DIR when set, else under sdl-agents root.
    """
    resolved, err = resolve_user_local_path(file_path)
    if err:
        return err
    assert resolved is not None
    pieces = to_langchain_documents(load_file(resolved))
    if not pieces:
        return f"No extractable content from {resolved}; check format."
    return "\n\n".join(doc.page_content for doc in pieces)


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant "
    "to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def generate_query_or_respond(state: MessagesState):
    response = _model().bind_tools([search_knowledge_base, load_document]).invoke(
        with_caveman(state["messages"])
    )
    return {"messages": [response]}


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = _grader().with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    if response.binary_score == "yes":
        return "generate_answer"
    return "rewrite_question"


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = _model().invoke(with_caveman([{"role": "user", "content": prompt}]))
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = _model().invoke(with_caveman([{"role": "user", "content": prompt}]))
    return {"messages": [response]}


def build_knowledge_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node(
        "retrieve",
        ToolNode([search_knowledge_base, load_document]),
    )
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    return workflow.compile()
