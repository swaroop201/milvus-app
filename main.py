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
from pymilvus.exceptions import DataNotMatchException
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Config — Zilliz Cloud or any Milvus (Docker/local)
# For Zilliz: set ZILLIZ_URI (cluster endpoint) and ZILLIZ_API_KEY, or use MILVUS_URI + MILVUS_TOKEN.
# Optional: MILVUS_USER + MILVUS_PASSWORD (used as token "user:password") instead of API key.
MILVUS_URI_RAW = os.environ.get("ZILLIZ_URI") or os.environ.get("MILVUS_URI", "http://localhost:19532")
MILVUS_TOKEN = os.environ.get("ZILLIZ_API_KEY") or os.environ.get("MILVUS_TOKEN", "")
MILVUS_USER = os.environ.get("MILVUS_USER", "").strip()
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "").strip()
# Zilliz serverless HTTPS without port can break gRPC; use 443 explicitly.
if (
    MILVUS_URI_RAW.startswith("https://")
    and "zillizcloud.com" in MILVUS_URI_RAW
    and ":443" not in MILVUS_URI_RAW
    and ":19530" not in MILVUS_URI_RAW
    and ":19540" not in MILVUS_URI_RAW
):
    MILVUS_URI = MILVUS_URI_RAW.rstrip("/") + ":443"
else:
    MILVUS_URI = MILVUS_URI_RAW
MILVUS_TIMEOUT = float(os.environ.get("MILVUS_TIMEOUT", "30"))
RAG_TOP_K = 5
# Only use retrieved chunks with similarity above this (COSINE: 0=far, 1=identical). Chunks below are ignored so the model can free-chat.
RAG_SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.5"))
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
    return os.environ.get("RAG_COLLECTION_NAME") or f"rag_docs_{get_embedding_dim()}"


def _rag_id_field() -> str:
    return os.environ.get("RAG_ID_FIELD", "id")


def _rag_text_field() -> str:
    return os.environ.get("RAG_TEXT_FIELD", "text")


def _rag_vector_field() -> str:
    return os.environ.get("RAG_VECTOR_FIELD", "embedding")


def _channel_name_max_length() -> int:
    return int(os.environ.get("RAG_CHANNEL_NAME_MAX_LENGTH", "512"))


def _format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for human-friendly timestamps."""
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_to_seconds(ts: str) -> int:
    """Parse HH:MM:SS (or MM:SS) to total seconds for YouTube &t=."""
    if not ts or not isinstance(ts, str):
        return 0
    ts = ts.strip()
    if not ts:
        return 0
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return int(parts[0]) if parts[0].isdigit() else 0
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0


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


def _get_milvus_token() -> str | None:
    """API key, or username:password for Zilliz if user/password are set."""
    if MILVUS_TOKEN:
        return MILVUS_TOKEN
    if MILVUS_USER and MILVUS_PASSWORD:
        return f"{MILVUS_USER}:{MILVUS_PASSWORD}"
    return None


def get_milvus_client() -> MilvusClient:
    """Create a new MilvusClient (fresh connection). Use timeout for Zilliz/serverless."""
    try:
        token = _get_milvus_token()
        kwargs = {"uri": MILVUS_URI, "timeout": MILVUS_TIMEOUT}
        if token:
            kwargs["token"] = token
        return MilvusClient(**kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Milvus at {MILVUS_URI}: {e}",
        )


def ensure_rag_collection(client: MilvusClient) -> None:
    """Create the RAG collection if it does not exist."""
    coll_name = get_rag_collection_name()
    dim = get_embedding_dim()
    for attempt in range(2):
        try:
            existing = client.list_collections()
            break
        except (ValueError, Exception) as e:
            if ("closed channel" in str(e).lower() or "inactive" in str(e).lower()) and attempt == 0:
                client = get_milvus_client()
                continue
            raise
    if coll_name in existing:
        return
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="channel_name", datatype=DataType.VARCHAR, max_length=512)
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
    channel_name: str | None = None


class TaskRequest(BaseModel):
    message: str


# --- Routes ---


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/hello")
def hello_world():
    return {"message": "Hello World"}


def _milvus_list_collections():
    """Call list_collections with one retry on closed channel (Zilliz serverless)."""
    client = get_milvus_client()
    try:
        return client.list_collections()
    except (ValueError, Exception) as e:
        err_msg = str(e).lower()
        if "closed channel" in err_msg or "inactive" in err_msg:
            client = get_milvus_client()
            return client.list_collections()
        raise


@app.get("/health/milvus")
def health_milvus():
    """Returns 200 if Milvus is reachable, 503 otherwise."""
    try:
        _milvus_list_collections()
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
    id_f = _rag_id_field()
    text_f = _rag_text_field()
    vector_f = _rag_vector_field()
    channel_f = "channel_name"
    max_ch = _channel_name_max_length()
    channel = (request.channel_name or "")[:max_ch]
    data_with_channel = [
        {id_f: i, text_f: t, vector_f: e, channel_f: channel}
        for i, t, e in zip(ids, chunks, embeddings)
    ]
    data_no_channel = [{id_f: i, text_f: t, vector_f: e} for i, t, e in zip(ids, chunks, embeddings)]
    try:
        milvus.insert(collection_name=coll_name, data=data_with_channel)
    except DataNotMatchException:
        milvus.insert(collection_name=coll_name, data=data_no_channel)
    return {"ingested": len(chunks), "message": f"Inserted {len(chunks)} chunks."}


def ingest_documents_with_metadata(
    texts: list[str],
    channel_names: list[str],
    video_ids: list[str],
    start_secs: list[float],
    end_secs: list[float],
) -> int:
    """
    Ingest document chunks (or segments) into the RAG collection with
    channel_name, video_id, and start/end timestamps per item.
    Used by ingest_transcripts_to_rag; all metadata lists must be aligned with texts.
    """
    if not texts:
        return 0
    milvus = get_milvus_client()
    ensure_rag_collection(milvus)
    coll_name = get_rag_collection_name()
    embeddings = embed_texts(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count mismatch.")
    ids = [abs(uuid.uuid4().int) % (2**63) for _ in texts]
    channels = channel_names
    vids = video_ids
    starts = start_secs
    ends = end_secs
    id_f = _rag_id_field()
    text_f = _rag_text_field()
    vector_f = _rag_vector_field()
    channel_f = "channel_name"
    max_ch = _channel_name_max_length()
    data = [
        {
            id_f: i,
            text_f: t,
            vector_f: e,
            channel_f: (c or "")[:max_ch],
            "video_id": (v or ""),
            "start_ts": _format_seconds_to_timestamp(float(s or 0.0)),
            "end_ts": _format_seconds_to_timestamp(float(en or 0.0)),
        }
        for i, t, e, c, v, s, en in zip(ids, texts, embeddings, channels, vids, starts, ends)
    ]
    data_no_channel = [{id_f: i, text_f: t, vector_f: e} for i, t, e in zip(ids, texts, embeddings)]
    try:
        milvus.insert(collection_name=coll_name, data=data)
    except DataNotMatchException:
        milvus.insert(collection_name=coll_name, data=data_no_channel)
    return len(texts)


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
    text_f = _rag_text_field()
    channel_f = "channel_name"
    try:
        results = milvus.search(
            collection_name=coll_name,
            data=[query_embedding],
            limit=RAG_TOP_K,
            output_fields=[text_f, channel_f, "$meta"],
        )
    except DataNotMatchException:
        results = milvus.search(
            collection_name=coll_name,
            data=[query_embedding],
            limit=RAG_TOP_K,
            output_fields=[text_f, "$meta"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Milvus search failed: {e}",
        )

    hits = results[0] if results else []
    context_parts: list[str] = []
    seen: set[str] = set()
    debug_lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        # Milvus COSINE returns similarity (higher=better). Only use chunks above threshold so we can free-chat when nothing matches.
        score = hit.get("distance")  # pymilvus often puts similarity in "distance" for COSINE
        if score is not None and isinstance(score, (int, float)):
            if score < RAG_SIMILARITY_THRESHOLD:
                continue
        entity = hit.get("entity") or hit
        text = entity.get(text_f)
        meta = entity.get("$meta") or {}
        if text and text not in seen:
            seen.add(text)
            channel_name = entity.get(channel_f) or ""
            # Dynamic fields may be stored either under $meta or flattened;
            # try both to be robust.
            video_id = meta.get("video_id") or entity.get("video_id") or ""
            start_ts = meta.get("start_ts") or entity.get("start_ts") or ""
            end_ts = meta.get("end_ts") or entity.get("end_ts") or ""
            # Collect a concise debug line for this retrieved chunk.
            if isinstance(score, (int, float)):
                score_str = f"{score:.3f}"
            else:
                score_str = "None"
            debug_lines.append(
                f"hit {idx}: score={score_str}, channel={channel_name!r}, "
                f"video_id={video_id}, start_ts={start_ts}, end_ts={end_ts}"
            )

            # Build a rich header with channel, URL (with &t= for start time), and timestamps for the LLM.
            header_parts: list[str] = []
            if channel_name:
                header_parts.append(f"Channel: {channel_name}")
            if video_id:
                # Include &t= so the link starts playback at this moment.
                start_sec = _timestamp_to_seconds(start_ts) if start_ts else 0
                if start_sec > 0:
                    url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}"
                else:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                header_parts.append(f"Video: {url}")
            if start_ts and end_ts:
                header_parts.append(f"Time: {start_ts}-{end_ts}")
            elif start_ts:
                header_parts.append(f"Time: {start_ts}")

            if header_parts:
                header_str = " | ".join(header_parts)
                context_parts.append(f"[{header_str}]\n{text}")
            else:
                context_parts.append(text)
    context = "\n\n".join(context_parts) if context_parts else ""
    has_relevant_context = bool(context)

    # Print debug information about which chunks were used for this query.
    if debug_lines:
        print(f"[RAG] Query: {query!r}")
        for line in debug_lines:
            print(f"[RAG]   {line}")

    system_content = (
        "You are a helpful, friendly assistant. Reply in a natural, conversational way.\n"
        "The retrieved context comes from OTHER people (for example YouTube creators like Ezachly, Joe Reis, Matthew Berman).\n"
        "The user is NOT the same person as these creators. Never assume the user runs their channel, has their salary, or said the statements in the context.\n"
        "When you use the context, talk about it explicitly, e.g. 'In Ezachly's video he says that…' or 'The transcripts suggest that…', and then give advice to the user in the second person ('you').\n"
        "Each context block may include a Channel and a Video URL. The Video URL may include &t=SECONDS so that clicking it starts the video at that moment. When you cite a source, use this format: put the full URL in parentheses and close the parenthesis immediately after the URL, then add the time range outside—e.g. (https://www.youtube.com/watch?v=WDwNow61JVE&t=330) 00:05:30-00:05:33. Never put the time range inside the parentheses.\n"
        "If 'Context from user's documents' is provided and it clearly answers the user's question, use it this way. If there is no context, or the user is greeting you ('how are you', 'hi'), making small talk, or asking something general (e.g. 'who is X', opinions, facts), answer from your own knowledge. Do NOT say 'there is no information in the context' or 'I can only confirm what is in the documents'. Just answer normally and briefly, as a person would in chat."
    )
    if has_relevant_context:
        user_content = f"Context from user's documents:\n{context}\n\nUser: {query}"
    else:
        user_content = f"User: {query}"

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
