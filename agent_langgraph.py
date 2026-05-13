"""
Agentic RAG with LangGraph — using Google Gemini instead of OpenAI.

Install dependencies:
    pip install -U langgraph "langchain[google-genai]" langchain-google-genai \
        langchain-community langchain-text-splitters bs4 \
        langchain-unstructured unstructured-client "unstructured[pdf]" python-magic
"""

import getpass
import os

from typing import Literal

from pydantic import BaseModel, Field

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_community.document_loaders import WebBaseLoader
# from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ← Gemini embeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.messages import convert_to_messages

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ── API key setup ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")


_set_env("GOOGLE_API_KEY")         # ← Gemini API key
# _set_env("UNSTRUCTURED_API_KEY")   # ← Unstructured API key (get one free at unstructured.io)


# ── 1. Preprocess documents ───────────────────────────────────────────────────
urls = [
    # "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    # "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://www.subtraction.com/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)


# ── 2. Create a retriever tool ────────────────────────────────────────────────
# Use Gemini embeddings instead of OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"),  # ← Gemini
)
retriever = vectorstore.as_retriever()


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about the indicated author's blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_blog_posts


# ── 3. Document loader tool ───────────────────────────────────────────────────
# @tool
# def load_document(file_path: str) -> str:
#     """Load and return the text content of a local file (PDF, DOCX, TXT, HTML,
#     images, and more) using the Unstructured loader. Pass an absolute or
#     relative path to the file you want to read."""
#     loader = UnstructuredLoader(
#         file_path=file_path,
#         api_key=os.getenv("UNSTRUCTURED_API_KEY"),
#         partition_via_api=True,
#     )
#     loaded_docs = loader.load()
#     if not loaded_docs:
#         return "No content could be extracted from the file."
#     return "\n\n".join(doc.page_content for doc in loaded_docs)


# document_loader_tool = load_document


# ── 3. Generate query or respond ──────────────────────────────────────────────
# Use Gemini 1.5 Pro instead of gpt-5.4
response_model = init_chat_model("gemini-3-flash", model_provider="google_genai", temperature=0)  # ← Gemini


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply
    respond to the user.
    """
    response = response_model.bind_tools([retriever_tool, document_loader_tool]).invoke(state["messages"])
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


grader_model = init_chat_model("gemini-1.5-pro", model_provider="google_genai", temperature=0)  # ← Gemini


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
    response = response_model.invoke([{"role": "user", "content": prompt}])
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
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ── 7. Assemble the graph ─────────────────────────────────────────────────────
workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
# workflow.add_node("retrieve", ToolNode([retriever_tool, document_loader_tool]))
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
                    "content": "How does Khoi Vinh feel about The Brutalist?",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")