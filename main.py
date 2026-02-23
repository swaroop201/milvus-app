import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Config
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19532")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
RAG_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chat: Groq (free) or OpenAI. Embeddings: Hugging Face (free, local) or OpenAI.
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_DIM = 384
HF_EMBEDDING_MODEL_INSTANCE = None  # Lazy load on first use
GROQ_MODEL = "llama-3.1-8b-instant"


def _use_groq() -> bool:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return bool(key)


def _use_hf_embeddings() -> bool:
    key = os.environ.get("HUGGINGFACE_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()
    return bool(key)


def get_embedding_dim() -> int:
    return HF_EMBEDDING_DIM if _use_hf_embeddings() else OPENAI_EMBEDDING_DIM


def get_rag_collection_name() -> str:
    return f"rag_docs_{get_embedding_dim()}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    groq = _use_groq()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    openai_ok = bool(openai_key) and openai_key != "your-openai-api-key-here"
    hf = _use_hf_embeddings()
    if groq:
        print("Chat: Groq (free) – GROQ_API_KEY is set.")
    elif openai_ok:
        print("Chat: OpenAI – OPENAI_API_KEY is set.")
    else:
        print("WARNING: No chat API key. Set GROQ_API_KEY (free) or OPENAI_API_KEY for /ask and /task.")
    if hf:
        print("Embeddings: Hugging Face (free) – HUGGINGFACE_TOKEN is set.")
    elif openai_ok:
        print("Embeddings: OpenAI – OPENAI_API_KEY is set.")
    else:
        print("WARNING: No embedding API. Set HUGGINGFACE_TOKEN (free) or OPENAI_API_KEY for /ingest and /task.")
    yield


app = FastAPI(lifespan=lifespan)

# Mount static files with cache control
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Optional: Add CORS middleware if needed for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_chat_client() -> OpenAI:
    """OpenAI-compatible client for chat: Groq (free) or OpenAI."""
    if _use_groq():
        key = os.environ.get("GROQ_API_KEY")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="Set GROQ_API_KEY (free at console.groq.com) or OPENAI_API_KEY for chat.",
        )
    return OpenAI(api_key=key)


def get_chat_model() -> str:
    return GROQ_MODEL if _use_groq() else "gpt-4o-mini"


def get_openai_client() -> OpenAI:
    """Only for OpenAI embeddings when not using HF."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY or HUGGINGFACE_TOKEN required for embeddings.",
        )
    return OpenAI(api_key=key)


def get_milvus_client() -> MilvusClient:
    try:
        if MILVUS_TOKEN:
            return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        return MilvusClient(uri=MILVUS_URI)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Milvus at {MILVUS_URI}: {e}",
        )


def ensure_rag_collection(client: MilvusClient) -> None:
    """Create the RAG collection if it does not exist."""
    coll_name = get_rag_collection_name()
    dim = get_embedding_dim()
    existing = client.list_collections()
    if coll_name in existing:
        return
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="AUTOINDEX",
    )
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=coll_name,
        schema=schema,
        index_params=index_params,
    )


def embed_texts_hf(texts: list[str]) -> list[list[float]]:
    """Get embeddings via sentence-transformers (local, free, no API key needed)."""
    global HF_EMBEDDING_MODEL_INSTANCE
    if HF_EMBEDDING_MODEL_INSTANCE is None:
        print(f"Loading embedding model: {HF_EMBEDDING_MODEL}")
        HF_EMBEDDING_MODEL_INSTANCE = SentenceTransformer(HF_EMBEDDING_MODEL)
    embeddings = HF_EMBEDDING_MODEL_INSTANCE.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings: Hugging Face (free) or OpenAI."""
    if not texts:
        return []
    if _use_hf_embeddings():
        return embed_texts_hf(texts)
    client = get_openai_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# --- Request models ---


class AskRequest(BaseModel):
    message: str


class IngestRequest(BaseModel):
    documents: list[str] | None = None
    text: str | None = None


class TaskRequest(BaseModel):
    message: str


# --- Routes ---


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/hello")
def hello_world():
    return {"message": "Hello World"}


@app.get("/health/milvus")
def health_milvus():
    """Returns 200 if Milvus is reachable, 503 otherwise."""
    try:
        client = get_milvus_client()
        client.list_collections()
        return {"milvus": "ok", "uri": MILVUS_URI}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Milvus unreachable: {e}")


@app.post("/ask")
def ask(request: AskRequest):
    client = get_chat_client()
    model = get_chat_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.message}],
        )
        reply = response.choices[0].message.content or ""
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/ingest")
def ingest(request: IngestRequest):
    """Ingest documents into Milvus for RAG. Send either `documents` (list of chunks) or `text` (will be chunked)."""
    if request.documents:
        chunks = [c.strip() for c in request.documents if c.strip()]
    elif request.text and request.text.strip():
        chunks = chunk_text(request.text)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'documents' (list of strings) or 'text' (string to chunk).",
        )
    if not chunks:
        return {"ingested": 0, "message": "No content to ingest."}

    milvus = get_milvus_client()
    ensure_rag_collection(milvus)
    embeddings = embed_texts(chunks)
    if len(embeddings) != len(chunks):
        raise HTTPException(status_code=502, detail="Embedding count mismatch.")

    ids = [abs(uuid.uuid4().int) % (2**63) for _ in chunks]
    coll_name = get_rag_collection_name()
    data = [
        {"id": i, "text": t, "embedding": e}
        for i, t, e in zip(ids, chunks, embeddings)
    ]
    milvus.insert(collection_name=coll_name, data=data)
    return {"ingested": len(chunks), "message": f"Inserted {len(chunks)} chunks."}


@app.post("/task")
def task(request: TaskRequest):
    """RAG endpoint: retrieve relevant chunks from Milvus and answer with chat API."""
    chat_client = get_chat_client()
    model = get_chat_model()
    milvus = get_milvus_client()
    ensure_rag_collection(milvus)

    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    [query_embedding] = embed_texts([query])
    coll_name = get_rag_collection_name()
    try:
        results = milvus.search(
            collection_name=coll_name,
            data=[query_embedding],
            limit=RAG_TOP_K,
            output_fields=["text"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Milvus search failed: {e}",
        )

    hits = results[0] if results else []
    context_parts = []
    seen = set()
    for hit in hits:
        entity = hit.get("entity") or hit
        text = entity.get("text")
        if text and text not in seen:
            seen.add(text)
            context_parts.append(text)
    context = "\n\n".join(context_parts) if context_parts else ""

    system_content = (
        "Answer the user's question using only the following context. "
        "If the context is empty or does not contain relevant information, say so. "
        "Do not make up facts."
    )
    user_content = (
        f"Context:\n{context}\n\nQuestion: {query}" if context else f"Question: {query}"
    )

    try:
        response = chat_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )
        reply = response.choices[0].message.content or ""
        return {"reply": reply, "retrieved": len(context_parts)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
