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

3. **Run Milvus** (required for RAG)

   **Option A — docker-compose** (recommended; uses etcd + MinIO, often more stable):
   ```bash
   docker-compose up -d
   ```
   This exposes Milvus on port **19532**. The app defaults to this port; override with `MILVUS_URI` in `.env` if needed.

   **Option B — single container:**
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.4.0-standalone
   ```
   Then set in `.env`: `MILVUS_URI=http://localhost:19530`.

   Optional: set `MILVUS_TOKEN` in `.env` if you use auth or a remote Milvus.

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
