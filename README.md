# FastAPI Hello World + RAG

A FastAPI app with a chat UI and **RAG** using **Milvus**. You can use **free APIs** (Groq + Hugging Face) or OpenAI.

## How to run the server

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys (free or paid)**

   **Free (recommended):**
   - **Chat:** Get a free key at [Groq Console](https://console.groq.com) → set `GROQ_API_KEY` in `.env`.
   - **Embeddings:** Get a free token at [Hugging Face → Settings → Tokens](https://huggingface.co/settings/tokens) → set `HUGGINGFACE_TOKEN` in `.env`.

   **Paid (optional):** Set `OPENAI_API_KEY` in `.env` to use OpenAI for chat and/or embeddings instead.  
   Copy `.env.example` to `.env`, then add your keys.

3. **Vector DB for RAG** (required for ingest/task)

   **Option A — Zilliz Cloud** (no Docker; recommended for production):
   - Create a cluster at [Zilliz Cloud](https://cloud.zilliz.com) and get your **cluster endpoint** (URI) and **API key**.
   - In `.env` set:
     - `ZILLIZ_URI=https://your-cluster-id.api.region.zillizcloud.com:19530`
     - `ZILLIZ_API_KEY=your-api-key`
   - Or use username/password: set `ZILLIZ_URI`, `MILVUS_USER`, and `MILVUS_PASSWORD` (the app sends them as `username:password` token).
   - **You do not create the collection or set dimensions** — the app creates the RAG collection automatically and uses **384** dims (Hugging Face) or **1536** dims (OpenAI) based on your embedding config.

   **Option B — Docker (local Milvus)**  
   **B1 — docker-compose** (etcd + MinIO):
   ```bash
   docker-compose up -d
   ```
   Exposes Milvus on port **19532**. Override with `MILVUS_URI` in `.env` if needed.

   **B2 — single container:**
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.4.0-standalone
   ```
   Then set `MILVUS_URI=http://localhost:19530` in `.env`.

   Optional: set `MILVUS_TOKEN` (or `ZILLIZ_API_KEY`) for auth.

4. **Start the server**

   ```bash
   python main.py
   ```

5. **Use the app**

   - **Chat UI:** **http://localhost:8000** — chat with ChatGPT (no RAG).
   - **Hello API:** **http://localhost:8000/api/hello**
   - **Ask:** `POST /ask` — `{"message": "..."}` → plain ChatGPT reply.
   - **Ingest (RAG):** `POST /ingest` — add documents to Milvus.
     - `{"documents": ["chunk1", "chunk2"]}` or `{"text": "long text to chunk"}`
   - **Task (RAG):** `POST /task` — `{"message": "your question"}` → answer using retrieved chunks + ChatGPT.
   - **Interactive docs:** **http://localhost:8000/docs**

The server runs on port 8000 by default.
