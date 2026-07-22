"""FastAPI application for an isolated, citation-aware PDF RAG service."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pypdf import PdfReader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
FASTEMBED_CACHE_DIR = Path(
    os.getenv("FASTEMBED_CACHE_DIR", DATA_DIR / "fastembed_cache")
)
STATIC_DIR = BASE_DIR / "static"
ALLOWED_TYPES = {"application/pdf", "application/x-pdf"}
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

SYSTEM_PROMPT = """You answer strictly from the supplied document context.
If the context is insufficient, say exactly: "The provided context does not contain sufficient information to answer this question."
Use conversation history only to understand follow-up questions; it is not a factual source.
Every factual claim must end with one or more citations in the exact form [filename, p. N].
Never invent a citation. Keep the answer concise and clear.

Conversation history:
{history}

Document context:
{context}

Question: {question}
Answer:"""

app = FastAPI(title="Document RAG API", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_embeddings: Any | None = None
_locks: dict[str, threading.Lock] = {}
_master_lock = threading.Lock()


class SessionCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=100)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class EvalCase(BaseModel):
    question: str
    expected_answer: str | None = None
    expected_pages: list[int] = Field(default_factory=list)


class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=50)


def safe_id(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise HTTPException(400, f"Invalid {label}")
    return value


def paths(user_id: str, session_id: str) -> dict[str, Path]:
    root = DATA_DIR / safe_id(user_id, "user id") / safe_id(session_id, "session id")
    return {
        "root": root,
        "uploads": root / "uploads",
        "chroma": root / "chroma",
        "history": root / "history.json",
        "meta": root / "session.json",
    }


def session_lock(user_id: str, session_id: str) -> threading.Lock:
    key = f"{user_id}/{session_id}"
    with _master_lock:
        return _locks.setdefault(key, threading.Lock())


class FastEmbedAdapter(Embeddings):
    """LangChain adapter for FastEmbed's lightweight ONNX embedding runtime."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        # FastEmbed otherwise uses the OS temporary directory. Windows can clean
        # that directory between the model download and ONNX initialization,
        # leaving a snapshot whose model.onnx file no longer exists.
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


def embeddings() -> Any:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedAdapter(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=FASTEMBED_CACHE_DIR,
        )
    return _embeddings


def vectorstore(p: dict[str, Path]) -> Chroma:
    return Chroma(
        collection_name="documents",
        persist_directory=str(p["chroma"]),
        embedding_function=embeddings(),
    )


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def require_session(user_id: str, session_id: str) -> dict[str, Path]:
    p = paths(user_id, session_id)
    if not p["meta"].exists():
        raise HTTPException(404, "Session not found")
    return p


def llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "GROQ_API_KEY is not configured")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=api_key,
        temperature=0.2,
    )


def retrieve(p: dict[str, Path], question: str) -> list[Document]:
    if not p["chroma"].exists():
        raise HTTPException(409, "Upload at least one PDF before asking a question")
    return vectorstore(p).similarity_search(question, k=TOP_K)


def context_for(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        page = int(doc.metadata.get("page", 0)) + 1
        source = doc.metadata.get("source_name", Path(str(doc.metadata.get("source", "document"))).name)
        blocks.append(f"SOURCE: [{source}, p. {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def citations_for(docs: list[Document]) -> list[dict]:
    seen, result = set(), []
    for doc in docs:
        source = doc.metadata.get("source_name", Path(str(doc.metadata.get("source", "document"))).name)
        page = int(doc.metadata.get("page", 0)) + 1
        key = (source, page)
        if key not in seen:
            seen.add(key)
            excerpt = " ".join(doc.page_content.split())[:240]
            result.append({"document": source, "page": page, "excerpt": excerpt})
    return result


def pdf_chunks(path: Path, source_name: str) -> tuple[list[Document], int]:
    """Extract page-aware overlapping chunks without importing the ML stack."""
    reader = PdfReader(str(path))
    chunks: list[Document] = []
    chunk_size, overlap = 1000, 200
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(Document(
                page_content=text[start:end],
                metadata={"source": str(path), "source_name": source_name, "page": page_index},
            ))
            if end == len(text):
                break
            start = end - overlap
    return chunks, len(reader.pages)


def answer_question(p: dict[str, Path], question: str, save: bool = True) -> dict:
    history = read_json(p["history"], [])
    docs = retrieve(p, question)
    compact_history = "\n".join(
        f"{item['role'].title()}: {item['content']}" for item in history[-10:]
    ) or "No previous messages."
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    messages = prompt.format_messages(
        history=compact_history, context=context_for(docs), question=question.strip()
    )
    answer = llm().invoke(messages).content
    citations = citations_for(docs)
    if save:
        history.extend([
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer, "citations": citations},
        ])
        write_json(p["history"], history)
    return {"answer": answer, "citations": citations, "retrieved_documents": docs}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/sessions", status_code=201)
def create_session(body: SessionCreate, x_user_id: Annotated[str, Header()]):
    user_id = safe_id(x_user_id, "user id")
    session_id = uuid.uuid4().hex
    p = paths(user_id, session_id)
    p["uploads"].mkdir(parents=True)
    write_json(p["meta"], {"id": session_id, "title": body.title, "documents": []})
    write_json(p["history"], [])
    return read_json(p["meta"], {})


@app.get("/api/sessions")
def list_sessions(x_user_id: Annotated[str, Header()]):
    root = DATA_DIR / safe_id(x_user_id, "user id")
    return [read_json(f, {}) for f in root.glob("*/session.json")] if root.exists() else []


@app.post("/api/sessions/{session_id}/documents")
async def upload_documents(
    session_id: str,
    files: Annotated[list[UploadFile], File()],
    x_user_id: Annotated[str, Header()],
):
    p = require_session(x_user_id, session_id)
    if not files:
        raise HTTPException(400, "No files supplied")
    all_splits, added = [], []
    with session_lock(x_user_id, session_id):
        for upload in files:
            name = Path(upload.filename or "document.pdf").name
            if upload.content_type not in ALLOWED_TYPES and not name.lower().endswith(".pdf"):
                raise HTTPException(415, f"{name} is not a PDF")
            content = await upload.read(MAX_FILE_MB * 1024 * 1024 + 1)
            if len(content) > MAX_FILE_MB * 1024 * 1024:
                raise HTTPException(413, f"{name} exceeds {MAX_FILE_MB} MB")
            stored = p["uploads"] / f"{uuid.uuid4().hex}_{name}"
            stored.write_bytes(content)
            try:
                splits, page_count = pdf_chunks(stored, name)
                all_splits.extend(splits)
                added.append({"name": name, "pages": page_count, "chunks": len(splits)})
            except Exception as exc:
                stored.unlink(missing_ok=True)
                raise HTTPException(422, f"Could not process {name}: {exc}") from exc
        if all_splits:
            vectorstore(p).add_documents(all_splits)
        meta = read_json(p["meta"], {})
        meta["documents"].extend(added)
        write_json(p["meta"], meta)
    return {"documents": added, "total_chunks": len(all_splits)}


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, body: ChatRequest, x_user_id: Annotated[str, Header()]):
    p = require_session(x_user_id, session_id)
    with session_lock(x_user_id, session_id):
        result = answer_question(p, body.question)
    result.pop("retrieved_documents")
    return result


@app.get("/api/sessions/{session_id}/history")
def history(session_id: str, x_user_id: Annotated[str, Header()]):
    p = require_session(x_user_id, session_id)
    return read_json(p["history"], [])


def token_f1(actual: str, expected: str) -> float:
    a, e = re.findall(r"\w+", actual.lower()), re.findall(r"\w+", expected.lower())
    common = sum((Counter(a) & Counter(e)).values())
    if not a or not e or not common:
        return 0.0
    precision, recall = common / len(a), common / len(e)
    return 2 * precision * recall / (precision + recall)


@app.post("/api/sessions/{session_id}/evaluate")
def evaluate(session_id: str, body: EvalRequest, x_user_id: Annotated[str, Header()]):
    p = require_session(x_user_id, session_id)
    results = []
    for case in body.cases:
        generated = answer_question(p, case.question, save=False)
        pages = {c["page"] for c in generated["citations"]}
        expected = set(case.expected_pages)
        results.append({
            "question": case.question,
            "citation_present": bool(re.search(r"\[[^\]]+, p\. \d+\]", generated["answer"])),
            "page_recall": len(pages & expected) / len(expected) if expected else None,
            "answer_token_f1": token_f1(generated["answer"], case.expected_answer) if case.expected_answer else None,
        })
    numeric_keys = ("citation_present", "page_recall", "answer_token_f1")
    summary = {}
    for key in numeric_keys:
        values = [float(r[key]) for r in results if r[key] is not None]
        summary[key] = round(sum(values) / len(values), 4) if values else None
    return {"summary": summary, "cases": results}


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, x_user_id: Annotated[str, Header()]):
    p = require_session(x_user_id, session_id)
    shutil.rmtree(p["root"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
